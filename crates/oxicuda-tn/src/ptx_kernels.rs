//! GPU PTX kernels for tensor network operations.
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

/// Naive tiled einsum `c[i,l] = sum_{j,k} a[i,j,k] * b[j,k,l]` for two 3-tensors.
///
/// Signature: `tensor_contract_kernel(a, b, c, n_i, n_j, n_k, n_l)`
/// Grid = (ceil(n_l/16), ceil(n_i/16), 1), Block = (16, 16, 1).
#[must_use]
pub fn tensor_contract_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry tensor_contract_kernel(\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u64 p_c,\n\
        .param .u32 p_n_i,\n\
        .param .u32 p_n_j,\n\
        .param .u32 p_n_k,\n\
        .param .u32 p_n_l\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_a];\n\
        ld.param.u64  %rd1, [p_b];\n\
        ld.param.u64  %rd2, [p_c];\n\
        ld.param.u32  %r0,  [p_n_i];\n\
        ld.param.u32  %r1,  [p_n_j];\n\
        ld.param.u32  %r2,  [p_n_k];\n\
        ld.param.u32  %r3,  [p_n_l];\n\
    \n\
        // i = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r4, %ntid.y;\n\
        mov.u32       %r5, %ctaid.y;\n\
        mov.u32       %r6, %tid.y;\n\
        mad.lo.u32    %r7, %r4, %r5, %r6;\n\
    \n\
        // l = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r8, %ntid.x;\n\
        mov.u32       %r9, %ctaid.x;\n\
        mov.u32       %r10, %tid.x;\n\
        mad.lo.u32    %r11, %r8, %r9, %r10;\n\
    \n\
        setp.ge.u32   %p0, %r7, %r0;\n\
        @%p0 bra $TC_DONE;\n\
        setp.ge.u32   %p0, %r11, %r3;\n\
        @%p0 bra $TC_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r12, 0;\n\
    \n\
    $TC_J:\n\
        setp.ge.u32   %p0, %r12, %r1;\n\
        @%p0 bra $TC_WRITE;\n\
    \n\
        mov.u32       %r13, 0;\n\
    \n\
    $TC_K:\n\
        setp.ge.u32   %p0, %r13, %r2;\n\
        @%p0 bra $TC_J_END;\n\
    \n\
        // a[i,j,k] = a[(i*n_j + j)*n_k + k]\n\
        mul.lo.u32    %r14, %r7, %r1;\n\
        add.u32       %r14, %r14, %r12;\n\
        mul.lo.u32    %r14, %r14, %r2;\n\
        add.u32       %r14, %r14, %r13;\n\
        mul.wide.u32  %rd3, %r14, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // b[j,k,l] = b[(j*n_k + k)*n_l + l]\n\
        mul.lo.u32    %r15, %r12, %r2;\n\
        add.u32       %r15, %r15, %r13;\n\
        mul.lo.u32    %r15, %r15, %r3;\n\
        add.u32       %r15, %r15, %r11;\n\
        mul.wide.u32  %rd5, %r15, 4;\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r13, %r13, 1;\n\
        bra $TC_K;\n\
    \n\
    $TC_J_END:\n\
        add.u32       %r12, %r12, 1;\n\
        bra $TC_J;\n\
    \n\
    $TC_WRITE:\n\
        // c[i, l] = c[i*n_l + l]\n\
        mul.lo.u32    %r16, %r7, %r3;\n\
        add.u32       %r16, %r16, %r11;\n\
        mul.wide.u32  %rd7, %r16, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $TC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Apply one Jacobi rotation pass over a 2-column block (Givens rotation).
///
/// Signature: `svd_jacobi_step_kernel(a, n_rows, n_cols, p, q, c, s)`
/// Block = (32, 1, 1). Rotates columns `p` and `q` by `[c, s; -s, c]`.
#[must_use]
pub fn svd_jacobi_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry svd_jacobi_step_kernel(\n\
        .param .u64 p_a,\n\
        .param .u32 p_n_rows,\n\
        .param .u32 p_n_cols,\n\
        .param .u32 p_p,\n\
        .param .u32 p_q,\n\
        .param .f32 p_c,\n\
        .param .f32 p_s\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_a];\n\
        ld.param.u32  %r0,  [p_n_rows];\n\
        ld.param.u32  %r1,  [p_n_cols];\n\
        ld.param.u32  %r2,  [p_p];\n\
        ld.param.u32  %r3,  [p_q];\n\
        ld.param.f32  %f0,  [p_c];\n\
        ld.param.f32  %f1,  [p_s];\n\
    \n\
        mov.u32       %r4, %ntid.x;\n\
        mov.u32       %r5, %ctaid.x;\n\
        mov.u32       %r6, %tid.x;\n\
        mad.lo.u32    %r7, %r4, %r5, %r6;\n\
    \n\
        setp.ge.u32   %p0, %r7, %r0;\n\
        @%p0 bra $SJ_DONE;\n\
    \n\
        // load a[row, p] = a[row*n_cols + p]\n\
        mul.lo.u32    %r8, %r7, %r1;\n\
        add.u32       %r8, %r8, %r2;\n\
        mul.wide.u32  %rd2, %r8, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd3];\n\
    \n\
        // load a[row, q] = a[row*n_cols + q]\n\
        mul.lo.u32    %r9, %r7, %r1;\n\
        add.u32       %r9, %r9, %r3;\n\
        mul.wide.u32  %rd4, %r9, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f3, [%rd5];\n\
    \n\
        // new_p =  c * a_p + s * a_q\n\
        mul.f32       %f4, %f0, %f2;\n\
        fma.rn.f32    %f4, %f1, %f3, %f4;\n\
        // new_q = -s * a_p + c * a_q\n\
        mul.f32       %f6, %f0, %f3;\n\
        neg.f32       %f7, %f1;\n\
        fma.rn.f32    %f5, %f7, %f2, %f6;\n\
    \n\
        st.global.f32 [%rd3], %f4;\n\
        st.global.f32 [%rd5], %f5;\n\
    \n\
    $SJ_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Apply local 2-site DMRG Hamiltonian to a 4-leg tensor.
///
/// Signature: `dmrg_local_apply_kernel(psi, h, out, d_l, d_p1, d_p2, d_r)`
/// out[a, p1, p2, b] = sum_{p1', p2'} h[p1, p2, p1', p2'] * psi[a, p1', p2', b]
#[must_use]
pub fn dmrg_local_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry dmrg_local_apply_kernel(\n\
        .param .u64 p_psi,\n\
        .param .u64 p_h,\n\
        .param .u64 p_out,\n\
        .param .u32 p_d_l,\n\
        .param .u32 p_d_p1,\n\
        .param .u32 p_d_p2,\n\
        .param .u32 p_d_r\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<32>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_psi];\n\
        ld.param.u64  %rd1, [p_h];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_d_l];\n\
        ld.param.u32  %r1,  [p_d_p1];\n\
        ld.param.u32  %r2,  [p_d_p2];\n\
        ld.param.u32  %r3,  [p_d_r];\n\
    \n\
        // composite index id = a*(d_p1*d_p2*d_r) + p1*(d_p2*d_r) + p2*d_r + b\n\
        mov.u32       %r4, %ntid.x;\n\
        mov.u32       %r5, %ctaid.x;\n\
        mov.u32       %r6, %tid.x;\n\
        mad.lo.u32    %r7, %r4, %r5, %r6;\n\
    \n\
        mul.lo.u32    %r8, %r1, %r2;\n\
        mul.lo.u32    %r8, %r8, %r3;     // d_p1*d_p2*d_r\n\
        mul.lo.u32    %r9, %r0, %r8;     // total elements\n\
        setp.ge.u32   %p0, %r7, %r9;\n\
        @%p0 bra $DM_DONE;\n\
    \n\
        // decode (a, p1, p2, b)\n\
        div.u32       %r10, %r7, %r8;    // a\n\
        rem.u32       %r11, %r7, %r8;    // rest\n\
        mul.lo.u32    %r12, %r2, %r3;    // d_p2*d_r\n\
        div.u32       %r13, %r11, %r12;  // p1\n\
        rem.u32       %r14, %r11, %r12;\n\
        div.u32       %r15, %r14, %r3;   // p2\n\
        rem.u32       %r16, %r14, %r3;   // b\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
    \n\
        mov.u32       %r17, 0;           // p1'\n\
    $DM_P1:\n\
        setp.ge.u32   %p0, %r17, %r1;\n\
        @%p0 bra $DM_WRITE;\n\
    \n\
        mov.u32       %r18, 0;           // p2'\n\
    $DM_P2:\n\
        setp.ge.u32   %p0, %r18, %r2;\n\
        @%p0 bra $DM_P1_END;\n\
    \n\
        // h[p1, p2, p1', p2'] index\n\
        mul.lo.u32    %r19, %r13, %r2;\n\
        add.u32       %r19, %r19, %r15;\n\
        mul.lo.u32    %r19, %r19, %r1;\n\
        add.u32       %r19, %r19, %r17;\n\
        mul.lo.u32    %r19, %r19, %r2;\n\
        add.u32       %r19, %r19, %r18;\n\
        mul.wide.u32  %rd3, %r19, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // psi[a, p1', p2', b] index\n\
        mul.lo.u32    %r20, %r10, %r1;\n\
        add.u32       %r20, %r20, %r17;\n\
        mul.lo.u32    %r20, %r20, %r2;\n\
        add.u32       %r20, %r20, %r18;\n\
        mul.lo.u32    %r20, %r20, %r3;\n\
        add.u32       %r20, %r20, %r16;\n\
        mul.wide.u32  %rd5, %r20, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r18, %r18, 1;\n\
        bra $DM_P2;\n\
    \n\
    $DM_P1_END:\n\
        add.u32       %r17, %r17, 1;\n\
        bra $DM_P1;\n\
    \n\
    $DM_WRITE:\n\
        mul.wide.u32  %rd7, %r7, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $DM_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Contract one MPO tensor over the physical leg of an MPS site tensor.
///
/// Signature: `mpo_apply_kernel(mps, mpo, out, dl, d, dr, wl, wr)`
/// out[(a,wl), p_out, (b,wr)] = sum_{p_in} mpo[wl, p_out, p_in, wr] * mps[a, p_in, b]
#[must_use]
pub fn mpo_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mpo_apply_kernel(\n\
        .param .u64 p_mps,\n\
        .param .u64 p_mpo,\n\
        .param .u64 p_out,\n\
        .param .u32 p_dl,\n\
        .param .u32 p_d,\n\
        .param .u32 p_dr,\n\
        .param .u32 p_wl,\n\
        .param .u32 p_wr\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<40>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_mps];\n\
        ld.param.u64  %rd1, [p_mpo];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_dl];\n\
        ld.param.u32  %r1,  [p_d];\n\
        ld.param.u32  %r2,  [p_dr];\n\
        ld.param.u32  %r3,  [p_wl];\n\
        ld.param.u32  %r4,  [p_wr];\n\
    \n\
        // gid = a*wl*d*b*wr + wl_*d*b*wr + p*b*wr + b*wr + wr_\n\
        mov.u32       %r5, %ntid.x;\n\
        mov.u32       %r6, %ctaid.x;\n\
        mov.u32       %r7, %tid.x;\n\
        mad.lo.u32    %r8, %r5, %r6, %r7;\n\
    \n\
        // total = dl*wl*d*dr*wr\n\
        mul.lo.u32    %r9, %r0, %r3;\n\
        mul.lo.u32    %r9, %r9, %r1;\n\
        mul.lo.u32    %r9, %r9, %r2;\n\
        mul.lo.u32    %r9, %r9, %r4;\n\
        setp.ge.u32   %p0, %r8, %r9;\n\
        @%p0 bra $MA_DONE;\n\
    \n\
        // decode (a, wl_, p, b, wr_) — row-major in the flat output\n\
        mov.u32       %r10, %r8;\n\
        // wr_\n\
        rem.u32       %r11, %r10, %r4;\n\
        div.u32       %r10, %r10, %r4;\n\
        // b\n\
        rem.u32       %r12, %r10, %r2;\n\
        div.u32       %r10, %r10, %r2;\n\
        // p\n\
        rem.u32       %r13, %r10, %r1;\n\
        div.u32       %r10, %r10, %r1;\n\
        // wl_\n\
        rem.u32       %r14, %r10, %r3;\n\
        // a\n\
        div.u32       %r15, %r10, %r3;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r16, 0;\n\
    \n\
    $MA_LOOP:\n\
        setp.ge.u32   %p0, %r16, %r1;\n\
        @%p0 bra $MA_WRITE;\n\
    \n\
        // mpo[wl_, p, p_in, wr_] = (((wl_*d)+p)*d + p_in)*wr + wr_\n\
        mul.lo.u32    %r17, %r14, %r1;\n\
        add.u32       %r17, %r17, %r13;\n\
        mul.lo.u32    %r17, %r17, %r1;\n\
        add.u32       %r17, %r17, %r16;\n\
        mul.lo.u32    %r17, %r17, %r4;\n\
        add.u32       %r17, %r17, %r11;\n\
        mul.wide.u32  %rd3, %r17, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // mps[a, p_in, b] = (a*d + p_in)*dr + b\n\
        mul.lo.u32    %r18, %r15, %r1;\n\
        add.u32       %r18, %r18, %r16;\n\
        mul.lo.u32    %r18, %r18, %r2;\n\
        add.u32       %r18, %r18, %r12;\n\
        mul.wide.u32  %rd5, %r18, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r16, %r16, 1;\n\
        bra $MA_LOOP;\n\
    \n\
    $MA_WRITE:\n\
        mul.wide.u32  %rd7, %r8, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $MA_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Apply a 4-leg Trotter gate `U[p1,p2,p1',p2']` to a two-site MPS block.
///
/// Signature: `trotter_step_kernel(theta, gate, out, dl, d, dr)`
/// out[a, p1, p2, b] = sum_{p1', p2'} gate[p1, p2, p1', p2'] * theta[a, p1', p2', b]
#[must_use]
pub fn trotter_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry trotter_step_kernel(\n\
        .param .u64 p_theta,\n\
        .param .u64 p_gate,\n\
        .param .u64 p_out,\n\
        .param .u32 p_dl,\n\
        .param .u32 p_d,\n\
        .param .u32 p_dr\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<32>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_theta];\n\
        ld.param.u64  %rd1, [p_gate];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_dl];\n\
        ld.param.u32  %r1,  [p_d];\n\
        ld.param.u32  %r2,  [p_dr];\n\
    \n\
        mov.u32       %r3, %ntid.x;\n\
        mov.u32       %r4, %ctaid.x;\n\
        mov.u32       %r5, %tid.x;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
    \n\
        // total = dl*d*d*dr\n\
        mul.lo.u32    %r7, %r0, %r1;\n\
        mul.lo.u32    %r7, %r7, %r1;\n\
        mul.lo.u32    %r7, %r7, %r2;\n\
        setp.ge.u32   %p0, %r6, %r7;\n\
        @%p0 bra $TS_DONE;\n\
    \n\
        // decode (a, p1, p2, b)\n\
        mov.u32       %r8, %r6;\n\
        rem.u32       %r9, %r8, %r2;\n\
        div.u32       %r8, %r8, %r2;        // b\n\
        rem.u32       %r10, %r8, %r1;\n\
        div.u32       %r8, %r8, %r1;        // p2\n\
        rem.u32       %r11, %r8, %r1;\n\
        div.u32       %r12, %r8, %r1;       // p1, then a\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r13, 0;\n\
    \n\
    $TS_P1:\n\
        setp.ge.u32   %p0, %r13, %r1;\n\
        @%p0 bra $TS_WRITE;\n\
    \n\
        mov.u32       %r14, 0;\n\
    $TS_P2:\n\
        setp.ge.u32   %p0, %r14, %r1;\n\
        @%p0 bra $TS_P1_END;\n\
    \n\
        // gate[p1, p2, p1', p2']\n\
        mul.lo.u32    %r15, %r11, %r1;\n\
        add.u32       %r15, %r15, %r10;\n\
        mul.lo.u32    %r15, %r15, %r1;\n\
        add.u32       %r15, %r15, %r13;\n\
        mul.lo.u32    %r15, %r15, %r1;\n\
        add.u32       %r15, %r15, %r14;\n\
        mul.wide.u32  %rd3, %r15, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // theta[a, p1', p2', b]\n\
        mul.lo.u32    %r16, %r12, %r1;\n\
        add.u32       %r16, %r16, %r13;\n\
        mul.lo.u32    %r16, %r16, %r1;\n\
        add.u32       %r16, %r16, %r14;\n\
        mul.lo.u32    %r16, %r16, %r2;\n\
        add.u32       %r16, %r16, %r9;\n\
        mul.wide.u32  %rd5, %r16, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r14, %r14, 1;\n\
        bra $TS_P2;\n\
    \n\
    $TS_P1_END:\n\
        add.u32       %r13, %r13, 1;\n\
        bra $TS_P1;\n\
    \n\
    $TS_WRITE:\n\
        mul.wide.u32  %rd7, %r6, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $TS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Mode-k unfolding of a 3-tensor into a matrix.
///
/// Signature: `hosvd_unfold_kernel(a, out, d0, d1, d2, mode)`
/// For mode=0: out[i, j*d2 + k] = a[i, j, k]; for mode=1: out[j, i*d2 + k]; for mode=2: out[k, i*d1 + j].
#[must_use]
pub fn hosvd_unfold_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry hosvd_unfold_kernel(\n\
        .param .u64 p_a,\n\
        .param .u64 p_out,\n\
        .param .u32 p_d0,\n\
        .param .u32 p_d1,\n\
        .param .u32 p_d2,\n\
        .param .u32 p_mode\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_a];\n\
        ld.param.u64  %rd1, [p_out];\n\
        ld.param.u32  %r0,  [p_d0];\n\
        ld.param.u32  %r1,  [p_d1];\n\
        ld.param.u32  %r2,  [p_d2];\n\
        ld.param.u32  %r3,  [p_mode];\n\
    \n\
        mov.u32       %r4, %ntid.x;\n\
        mov.u32       %r5, %ctaid.x;\n\
        mov.u32       %r6, %tid.x;\n\
        mad.lo.u32    %r7, %r4, %r5, %r6;\n\
    \n\
        // total = d0*d1*d2\n\
        mul.lo.u32    %r8, %r0, %r1;\n\
        mul.lo.u32    %r8, %r8, %r2;\n\
        setp.ge.u32   %p0, %r7, %r8;\n\
        @%p0 bra $UF_DONE;\n\
    \n\
        // decode (i, j, k)\n\
        rem.u32       %r9, %r7, %r2;          // k\n\
        div.u32       %r10, %r7, %r2;\n\
        rem.u32       %r11, %r10, %r1;        // j\n\
        div.u32       %r12, %r10, %r1;        // i\n\
    \n\
        // load a[i,j,k]\n\
        mul.wide.u32  %rd2, %r7, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        // out_index per mode\n\
        setp.ne.u32   %p0, %r3, 0;\n\
        @%p0 bra $UF_M1;\n\
    \n\
        // mode 0: out[i*(d1*d2) + j*d2 + k]\n\
        mul.lo.u32    %r13, %r1, %r2;\n\
        mul.lo.u32    %r14, %r12, %r13;\n\
        mul.lo.u32    %r15, %r11, %r2;\n\
        add.u32       %r16, %r14, %r15;\n\
        add.u32       %r16, %r16, %r9;\n\
        bra $UF_STORE;\n\
    \n\
    $UF_M1:\n\
        setp.ne.u32   %p0, %r3, 1;\n\
        @%p0 bra $UF_M2;\n\
    \n\
        // mode 1: out[j*(d0*d2) + i*d2 + k]\n\
        mul.lo.u32    %r13, %r0, %r2;\n\
        mul.lo.u32    %r14, %r11, %r13;\n\
        mul.lo.u32    %r15, %r12, %r2;\n\
        add.u32       %r16, %r14, %r15;\n\
        add.u32       %r16, %r16, %r9;\n\
        bra $UF_STORE;\n\
    \n\
    $UF_M2:\n\
        // mode 2: out[k*(d0*d1) + i*d1 + j]\n\
        mul.lo.u32    %r13, %r0, %r1;\n\
        mul.lo.u32    %r14, %r9, %r13;\n\
        mul.lo.u32    %r15, %r12, %r1;\n\
        add.u32       %r16, %r14, %r15;\n\
        add.u32       %r16, %r16, %r11;\n\
    \n\
    $UF_STORE:\n\
        mul.wide.u32  %rd4, %r16, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        st.global.f32 [%rd5], %f0;\n\
    \n\
    $UF_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// TT-rounding pass: copy a TT core for a left-right sweep step.
///
/// Signature: `tt_round_kernel(core_in, core_out, r_l, n, r_r)`
/// Identity copy for round; SVD update happens host-side after.
#[must_use]
pub fn tt_round_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry tt_round_kernel(\n\
        .param .u64 p_core_in,\n\
        .param .u64 p_core_out,\n\
        .param .u32 p_r_l,\n\
        .param .u32 p_n,\n\
        .param .u32 p_r_r\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<2>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_core_in];\n\
        ld.param.u64  %rd1, [p_core_out];\n\
        ld.param.u32  %r0,  [p_r_l];\n\
        ld.param.u32  %r1,  [p_n];\n\
        ld.param.u32  %r2,  [p_r_r];\n\
    \n\
        mov.u32       %r3, %ntid.x;\n\
        mov.u32       %r4, %ctaid.x;\n\
        mov.u32       %r5, %tid.x;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
    \n\
        mul.lo.u32    %r7, %r0, %r1;\n\
        mul.lo.u32    %r7, %r7, %r2;\n\
        setp.ge.u32   %p0, %r6, %r7;\n\
        @%p0 bra $TR_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r6, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
        add.u64       %rd4, %rd1, %rd2;\n\
        st.global.f32 [%rd4], %f0;\n\
    \n\
    $TR_DONE:\n\
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
            ("tensor_contract", tensor_contract_ptx),
            ("svd_jacobi_step", svd_jacobi_step_ptx),
            ("dmrg_local_apply", dmrg_local_apply_ptx),
            ("mpo_apply", mpo_apply_ptx),
            ("trotter_step", trotter_step_ptx),
            ("hosvd_unfold", hosvd_unfold_ptx),
            ("tt_round", tt_round_ptx),
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
}
