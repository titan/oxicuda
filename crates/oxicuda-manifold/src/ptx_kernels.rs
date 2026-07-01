//! GPU PTX kernels for manifold learning operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM>=100 -> 8.7 (Blackwell), SM>=90 -> 8.4 (Hopper),
//!     SM>=80  -> 8.0 (Ampere),    else  -> 7.5 (Turing).
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

/// Pairwise squared Euclidean distance: `d[i,j] = sum_k (x[i,k] - x[j,k])^2`.
///
/// Signature: `pairwise_dist_sq_kernel(x, d, n, dim)`
/// Grid = (ceil(n/16), ceil(n/16), 1), Block = (16, 16, 1).
#[must_use]
pub fn pairwise_dist_sq_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry pairwise_dist_sq_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_d,\n\
        .param .u32 p_n,\n\
        .param .u32 p_dim\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_d];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_dim];\n\
    \n\
        mov.u32       %r2, %ntid.y;\n\
        mov.u32       %r3, %ctaid.y;\n\
        mov.u32       %r4, %tid.y;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        mov.u32       %r6, %ntid.x;\n\
        mov.u32       %r7, %ctaid.x;\n\
        mov.u32       %r8, %tid.x;\n\
        mad.lo.u32    %r9, %r6, %r7, %r8;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $PD_DONE;\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $PD_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r10, 0;\n\
    \n\
    $PD_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r1;\n\
        @%p0 bra $PD_WRITE;\n\
    \n\
        // x[i, k]\n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        // x[j, k]\n\
        mul.lo.u32    %r12, %r9, %r1;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd4, %r12, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f2, [%rd5];\n\
    \n\
        sub.f32       %f3, %f1, %f2;\n\
        fma.rn.f32    %f0, %f3, %f3, %f0;\n\
    \n\
        add.u32       %r10, %r10, 1;\n\
        bra $PD_LOOP;\n\
    \n\
    $PD_WRITE:\n\
        mul.lo.u32    %r13, %r5, %r0;\n\
        add.u32       %r13, %r13, %r9;\n\
        mul.wide.u32  %rd6, %r13, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.f32 [%rd7], %f0;\n\
    \n\
    $PD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Per-row top-k smallest neighbours of a precomputed distance matrix.
///
/// Signature: `knn_topk_kernel(d, idx, dist_out, n, k)`
/// Each thread processes one row, maintaining an ascending-sorted top-k buffer
/// (`dist_out[row, 0..k]`) and matching column indices (`idx[row, 0..k]`). A
/// candidate `d[row, j]` (`j != row`) that beats the current worst (slot `k-1`)
/// is written into the last slot and bubbled up while it is smaller than its
/// predecessor — a genuine insertion sort (k is small).
#[must_use]
pub fn knn_topk_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry knn_topk_kernel(\n\
        .param .u64 p_d,\n\
        .param .u64 p_idx,\n\
        .param .u64 p_dist_out,\n\
        .param .u32 p_n,\n\
        .param .u32 p_k\n\
    )\n\
    {\n\
        .reg .u64  %rd<20>;\n\
        .reg .u32  %r<32>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_d];\n\
        ld.param.u64  %rd1, [p_idx];\n\
        ld.param.u64  %rd2, [p_dist_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_k];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $KT_DONE;\n\
    \n\
        // base = row * k  (start of this row's top-k block)\n\
        mul.lo.u32    %r20, %r5, %r1;\n\
    \n\
        // Initialise top-k buffer to +inf, indices to 0\n\
        mov.u32       %r6, 0;\n\
    $KT_INIT:\n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $KT_SCAN;\n\
        add.u32       %r7, %r20, %r6;\n\
        mul.wide.u32  %rd3, %r7, 4;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        mov.f32       %f0, 0f7F800000;\n\
        st.global.f32 [%rd4], %f0;\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        mov.u32       %r8, 0;\n\
        st.global.u32 [%rd5], %r8;\n\
        add.u32       %r6, %r6, 1;\n\
        bra $KT_INIT;\n\
    \n\
    $KT_SCAN:\n\
        mov.u32       %r9, 0;\n\
    $KT_OUTER:\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $KT_DONE;\n\
        setp.eq.u32   %p1, %r9, %r5;\n\
        @%p1 bra $KT_NEXT;\n\
    \n\
        // load d[row, j]\n\
        mul.lo.u32    %r10, %r5, %r0;\n\
        add.u32       %r10, %r10, %r9;\n\
        mul.wide.u32  %rd6, %r10, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f1, [%rd7];\n\
    \n\
        // worst = dist_out[row, k-1]\n\
        sub.u32       %r11, %r1, 1;\n\
        add.u32       %r12, %r20, %r11;\n\
        mul.wide.u32  %rd8, %r12, 4;\n\
        add.u64       %rd9, %rd2, %rd8;\n\
        ld.global.f32 %f2, [%rd9];\n\
    \n\
        setp.ge.f32   %p1, %f1, %f2;\n\
        @%p1 bra $KT_NEXT;\n\
    \n\
        // insert candidate into the last (worst) slot\n\
        st.global.f32 [%rd9], %f1;\n\
        add.u64       %rd10, %rd1, %rd8;\n\
        st.global.u32 [%rd10], %r9;\n\
    \n\
        // bubble up: p = k-1; while p>0 and dist[p] < dist[p-1] swap\n\
        mov.u32       %r13, %r11;\n\
    $KT_BUB:\n\
        setp.eq.u32   %p0, %r13, 0;\n\
        @%p0 bra $KT_NEXT;\n\
        add.u32       %r14, %r20, %r13;\n\
        sub.u32       %r15, %r13, 1;\n\
        add.u32       %r16, %r20, %r15;\n\
        mul.wide.u32  %rd11, %r14, 4;\n\
        add.u64       %rd12, %rd2, %rd11;\n\
        mul.wide.u32  %rd13, %r16, 4;\n\
        add.u64       %rd14, %rd2, %rd13;\n\
        ld.global.f32 %f3, [%rd12];\n\
        ld.global.f32 %f4, [%rd14];\n\
        setp.ge.f32   %p0, %f3, %f4;\n\
        @%p0 bra $KT_NEXT;\n\
        // swap distances\n\
        st.global.f32 [%rd12], %f4;\n\
        st.global.f32 [%rd14], %f3;\n\
        // swap indices\n\
        add.u64       %rd15, %rd1, %rd11;\n\
        add.u64       %rd16, %rd1, %rd13;\n\
        ld.global.u32 %r17, [%rd15];\n\
        ld.global.u32 %r18, [%rd16];\n\
        st.global.u32 [%rd15], %r18;\n\
        st.global.u32 [%rd16], %r17;\n\
        sub.u32       %r13, %r13, 1;\n\
        bra $KT_BUB;\n\
    \n\
    $KT_NEXT:\n\
        add.u32       %r9, %r9, 1;\n\
        bra $KT_OUTER;\n\
    \n\
    $KT_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// t-SNE attractive/repulsive gradient step: per-pair update on Y.
///
/// Signature: `tsne_grad_kernel(p, q, y, grad, n, dim)`
/// `grad[i] = sum_j (p[i,j] - q[i,j]) * q[i,j] * (y[i] - y[j])`
#[must_use]
pub fn tsne_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry tsne_grad_kernel(\n\
        .param .u64 p_p,\n\
        .param .u64 p_q,\n\
        .param .u64 p_y,\n\
        .param .u64 p_grad,\n\
        .param .u32 p_n,\n\
        .param .u32 p_dim\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<14>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_p];\n\
        ld.param.u64  %rd1, [p_q];\n\
        ld.param.u64  %rd2, [p_y];\n\
        ld.param.u64  %rd3, [p_grad];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_dim];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $TG_DONE;\n\
    \n\
        // For each output dim d, accumulate sum_j (p_ij - q_ij)*q_ij*(y_id - y_jd)\n\
        mov.u32       %r6, 0;\n\
    $TG_DIM:\n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $TG_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        // y[i, d]\n\
        mul.lo.u32    %r7, %r5, %r1;\n\
        add.u32       %r7, %r7, %r6;\n\
        mul.wide.u32  %rd4, %r7, 4;\n\
        add.u64       %rd5, %rd2, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        mov.u32       %r8, 0;\n\
    $TG_J:\n\
        setp.ge.u32   %p0, %r8, %r0;\n\
        @%p0 bra $TG_WRITE;\n\
        setp.eq.u32   %p1, %r8, %r5;\n\
        @%p1 bra $TG_J_NEXT;\n\
    \n\
        // p[i, j]\n\
        mul.lo.u32    %r9, %r5, %r0;\n\
        add.u32       %r9, %r9, %r8;\n\
        mul.wide.u32  %rd6, %r9, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
    \n\
        // q[i, j]\n\
        add.u64       %rd8, %rd1, %rd6;\n\
        ld.global.f32 %f3, [%rd8];\n\
    \n\
        // y[j, d]\n\
        mul.lo.u32    %r10, %r8, %r1;\n\
        add.u32       %r10, %r10, %r6;\n\
        mul.wide.u32  %rd9, %r10, 4;\n\
        add.u64       %rd10, %rd2, %rd9;\n\
        ld.global.f32 %f4, [%rd10];\n\
    \n\
        sub.f32       %f5, %f2, %f3;\n\
        mul.f32       %f6, %f5, %f3;\n\
        sub.f32       %f7, %f1, %f4;\n\
        fma.rn.f32    %f0, %f6, %f7, %f0;\n\
    \n\
    $TG_J_NEXT:\n\
        add.u32       %r8, %r8, 1;\n\
        bra $TG_J;\n\
    \n\
    $TG_WRITE:\n\
        // grad[i, d] = 4 * f0\n\
        mov.f32       %f8, 0f40800000;\n\
        mul.f32       %f0, %f0, %f8;\n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r6;\n\
        mul.wide.u32  %rd11, %r11, 4;\n\
        add.u64       %rd12, %rd3, %rd11;\n\
        st.global.f32 [%rd12], %f0;\n\
    \n\
        add.u32       %r6, %r6, 1;\n\
        bra $TG_DIM;\n\
    \n\
    $TG_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// UMAP edge-wise SGD step: pulls connected pairs together and pushes negatives apart.
///
/// Signature: `umap_step_kernel(y, edges_i, edges_j, n_edges, dim, alpha)`
/// `y[i] += alpha * grad_attract(y_i, y_j); y[j] -= alpha * grad_attract(y_i, y_j)`
#[must_use]
pub fn umap_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry umap_step_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_ei,\n\
        .param .u64 p_ej,\n\
        .param .u32 p_n_edges,\n\
        .param .u32 p_dim,\n\
        .param .f32 p_alpha\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<12>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_ei];\n\
        ld.param.u64  %rd2, [p_ej];\n\
        ld.param.u32  %r0,  [p_n_edges];\n\
        ld.param.u32  %r1,  [p_dim];\n\
        ld.param.f32  %f0,  [p_alpha];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $US_DONE;\n\
    \n\
        // load edge endpoints\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.u32 %r6, [%rd4];\n\
        add.u64       %rd5, %rd2, %rd3;\n\
        ld.global.u32 %r7, [%rd5];\n\
    \n\
        // ||y_i - y_j||^2\n\
        mov.f32       %f1, 0f00000000;\n\
        mov.u32       %r8, 0;\n\
    $US_NORM:\n\
        setp.ge.u32   %p0, %r8, %r1;\n\
        @%p0 bra $US_UPDATE;\n\
        mul.lo.u32    %r9, %r6, %r1;\n\
        add.u32       %r9, %r9, %r8;\n\
        mul.wide.u32  %rd6, %r9, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
        mul.lo.u32    %r10, %r7, %r1;\n\
        add.u32       %r10, %r10, %r8;\n\
        mul.wide.u32  %rd8, %r10, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f3, [%rd9];\n\
        sub.f32       %f4, %f2, %f3;\n\
        fma.rn.f32    %f1, %f4, %f4, %f1;\n\
        add.u32       %r8, %r8, 1;\n\
        bra $US_NORM;\n\
    \n\
    $US_UPDATE:\n\
        // coef = -2*alpha / (1 + dist^2)\n\
        mov.f32       %f5, 0f3F800000;\n\
        add.f32       %f5, %f5, %f1;\n\
        mov.f32       %f6, 0f40000000;\n\
        mul.f32       %f6, %f6, %f0;\n\
        neg.f32       %f6, %f6;\n\
        div.rn.f32    %f7, %f6, %f5;\n\
    \n\
        mov.u32       %r8, 0;\n\
    $US_APPLY:\n\
        setp.ge.u32   %p0, %r8, %r1;\n\
        @%p0 bra $US_DONE;\n\
        mul.lo.u32    %r9, %r6, %r1;\n\
        add.u32       %r9, %r9, %r8;\n\
        mul.wide.u32  %rd6, %r9, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
        mul.lo.u32    %r10, %r7, %r1;\n\
        add.u32       %r10, %r10, %r8;\n\
        mul.wide.u32  %rd8, %r10, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f3, [%rd9];\n\
        sub.f32       %f4, %f2, %f3;\n\
        mul.f32       %f8, %f7, %f4;\n\
        add.f32       %f2, %f2, %f8;\n\
        sub.f32       %f3, %f3, %f8;\n\
        st.global.f32 [%rd7], %f2;\n\
        st.global.f32 [%rd9], %f3;\n\
        add.u32       %r8, %r8, 1;\n\
        bra $US_APPLY;\n\
    \n\
    $US_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// PCA centering: subtract column means from each row.
///
/// Signature: `pca_center_kernel(x, mean, n, dim)`
/// `x[i, d] -= mean[d]`
#[must_use]
pub fn pca_center_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry pca_center_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_mean,\n\
        .param .u32 p_n,\n\
        .param .u32 p_dim\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_mean];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_dim];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        mul.lo.u32    %r6, %r0, %r1;\n\
        setp.ge.u32   %p0, %r5, %r6;\n\
        @%p0 bra $PC_DONE;\n\
    \n\
        rem.u32       %r7, %r5, %r1;\n\
        mul.wide.u32  %rd2, %r5, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
        mul.wide.u32  %rd4, %r7, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        sub.f32       %f2, %f0, %f1;\n\
        st.global.f32 [%rd3], %f2;\n\
    \n\
    $PC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// MDS double centering: B = -0.5 * J * D^2 * J where J = I - (1/n)11^T.
///
/// Signature: `mds_double_center_kernel(d2, row_mean, col_mean, total_mean, b, n)`
/// `b[i,j] = -0.5*(d2[i,j] - row_mean[i] - col_mean[j] + total_mean)`
#[must_use]
pub fn mds_double_center_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mds_double_center_kernel(\n\
        .param .u64 p_d2,\n\
        .param .u64 p_row_mean,\n\
        .param .u64 p_col_mean,\n\
        .param .f32 p_total_mean,\n\
        .param .u64 p_b,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_d2];\n\
        ld.param.u64  %rd1, [p_row_mean];\n\
        ld.param.u64  %rd2, [p_col_mean];\n\
        ld.param.f32  %f0,  [p_total_mean];\n\
        ld.param.u64  %rd3, [p_b];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.y;\n\
        mov.u32       %r2, %ctaid.y;\n\
        mov.u32       %r3, %tid.y;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        mov.u32       %r5, %ntid.x;\n\
        mov.u32       %r6, %ctaid.x;\n\
        mov.u32       %r7, %tid.x;\n\
        mad.lo.u32    %r8, %r5, %r6, %r7;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $MC_DONE;\n\
        setp.ge.u32   %p0, %r8, %r0;\n\
        @%p0 bra $MC_DONE;\n\
    \n\
        mul.lo.u32    %r9, %r4, %r0;\n\
        add.u32       %r9, %r9, %r8;\n\
        mul.wide.u32  %rd4, %r9, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
    \n\
        mul.wide.u32  %rd8, %r8, 4;\n\
        add.u64       %rd9, %rd2, %rd8;\n\
        ld.global.f32 %f3, [%rd9];\n\
    \n\
        sub.f32       %f4, %f1, %f2;\n\
        sub.f32       %f4, %f4, %f3;\n\
        add.f32       %f4, %f4, %f0;\n\
        mov.f32       %f5, 0fBF000000;\n\
        mul.f32       %f4, %f4, %f5;\n\
    \n\
        add.u64       %rd10, %rd3, %rd4;\n\
        st.global.f32 [%rd10], %f4;\n\
    \n\
    $MC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Johnson-Lindenstrauss random projection: out[i, k] = sum_d x[i, d] * R[d, k].
///
/// Signature: `random_proj_kernel(x, r, out, n, d, k)`
#[must_use]
pub fn random_proj_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry random_proj_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_r,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n,\n\
        .param .u32 p_d,\n\
        .param .u32 p_k\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_r];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_d];\n\
        ld.param.u32  %r2,  [p_k];\n\
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
        setp.ge.u32   %p0, %r6, %r0;\n\
        @%p0 bra $RP_DONE;\n\
        setp.ge.u32   %p0, %r10, %r2;\n\
        @%p0 bra $RP_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r11, 0;\n\
    \n\
    $RP_LOOP:\n\
        setp.ge.u32   %p0, %r11, %r1;\n\
        @%p0 bra $RP_WRITE;\n\
    \n\
        mul.lo.u32    %r12, %r6, %r1;\n\
        add.u32       %r12, %r12, %r11;\n\
        mul.wide.u32  %rd3, %r12, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        mul.lo.u32    %r13, %r11, %r2;\n\
        add.u32       %r13, %r13, %r10;\n\
        mul.wide.u32  %rd5, %r13, 4;\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r11, %r11, 1;\n\
        bra $RP_LOOP;\n\
    \n\
    $RP_WRITE:\n\
        mul.lo.u32    %r14, %r6, %r2;\n\
        add.u32       %r14, %r14, %r10;\n\
        mul.wide.u32  %rd7, %r14, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $RP_DONE:\n\
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
            ("pairwise_dist_sq", pairwise_dist_sq_ptx),
            ("knn_topk", knn_topk_ptx),
            ("tsne_grad", tsne_grad_ptx),
            ("umap_step", umap_step_ptx),
            ("pca_center", pca_center_ptx),
            ("mds_double_center", mds_double_center_ptx),
            ("random_proj", random_proj_ptx),
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
                assert!(!s.is_empty(), "kernel {name} sm={sm} empty");
                assert!(
                    s.contains(".visible .entry"),
                    "kernel {name} sm={sm} missing entry"
                );
                assert!(s.contains("ret"), "kernel {name} sm={sm} missing ret");
            }
        }
    }
}
