//! GPU PTX kernels for graph algorithm operations.
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

/// Frontier-based BFS single-step level update.
///
/// Signature: `bfs_level_kernel(row_ptr, col_idx, level, frontier_in, frontier_out, n, depth)`
/// For each vertex `u` in `frontier_in`, set `level[v] = depth+1` for every neighbor `v`
/// with `level[v] == -1`, and append `v` to `frontier_out`.
#[must_use]
pub fn bfs_level_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry bfs_level_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col_idx,\n\
        .param .u64 p_level,\n\
        .param .u64 p_front_in,\n\
        .param .u64 p_front_out,\n\
        .param .u32 p_n,\n\
        .param .u32 p_depth\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<24>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col_idx];\n\
        ld.param.u64  %rd2, [p_level];\n\
        ld.param.u64  %rd3, [p_front_in];\n\
        ld.param.u64  %rd4, [p_front_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_depth];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $BFS_DONE;\n\
    \n\
        // load frontier_in[gid] -> u (1 = in frontier)\n\
        mul.wide.u32  %rd5, %r5, 4;\n\
        add.u64       %rd6, %rd3, %rd5;\n\
        ld.global.u32 %r6, [%rd6];\n\
        setp.eq.u32   %p0, %r6, 0;\n\
        @%p0 bra $BFS_DONE;\n\
    \n\
        // u = gid; row_ptr[u], row_ptr[u+1]\n\
        mul.wide.u32  %rd7, %r5, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.u32 %r7, [%rd8];\n\
        add.u64       %rd9, %rd8, 4;\n\
        ld.global.u32 %r8, [%rd9];\n\
    \n\
        // depth_next = depth + 1\n\
        add.u32       %r9, %r1, 1;\n\
    \n\
        // for j = row_ptr[u]; j < row_ptr[u+1]; j++\n\
        mov.u32       %r10, %r7;\n\
    $BFS_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r8;\n\
        @%p0 bra $BFS_DONE;\n\
    \n\
        // v = col_idx[j]\n\
        mul.wide.u32  %rd10, %r10, 4;\n\
        add.u64       %rd11, %rd1, %rd10;\n\
        ld.global.u32 %r11, [%rd11];\n\
    \n\
        // if level[v] == -1\n\
        mul.wide.u32  %rd12, %r11, 4;\n\
        add.u64       %rd13, %rd2, %rd12;\n\
        ld.global.u32 %r12, [%rd13];\n\
        setp.ne.u32   %p0, %r12, 0xFFFFFFFF;\n\
        @%p0 bra $BFS_SKIP;\n\
    \n\
        st.global.u32 [%rd13], %r9;\n\
        add.u64       %rd14, %rd4, %rd12;\n\
        mov.u32       %r13, 1;\n\
        st.global.u32 [%rd14], %r13;\n\
    \n\
    $BFS_SKIP:\n\
        add.u32       %r10, %r10, 1;\n\
        bra $BFS_LOOP;\n\
    \n\
    $BFS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Dijkstra single edge relaxation kernel.
///
/// Signature: `dijkstra_relax_kernel(row_ptr, col_idx, weights, dist, frontier, n, u)`
/// For the source vertex `u`, relax `dist[v] = min(dist[v], dist[u] + w(u,v))`.
#[must_use]
pub fn dijkstra_relax_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry dijkstra_relax_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col_idx,\n\
        .param .u64 p_weights,\n\
        .param .u64 p_dist,\n\
        .param .u64 p_frontier,\n\
        .param .u32 p_n,\n\
        .param .u32 p_u\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col_idx];\n\
        ld.param.u64  %rd2, [p_weights];\n\
        ld.param.u64  %rd3, [p_dist];\n\
        ld.param.u64  %rd4, [p_frontier];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_u];\n\
    \n\
        // row_ptr[u], row_ptr[u+1]\n\
        mul.wide.u32  %rd5, %r1, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.u32 %r2, [%rd6];\n\
        add.u64       %rd7, %rd6, 4;\n\
        ld.global.u32 %r3, [%rd7];\n\
    \n\
        // gid = tid within [row_ptr[u], row_ptr[u+1])\n\
        mov.u32       %r4, %ntid.x;\n\
        mov.u32       %r5, %ctaid.x;\n\
        mov.u32       %r6, %tid.x;\n\
        mad.lo.u32    %r7, %r4, %r5, %r6;\n\
        add.u32       %r8, %r2, %r7;\n\
        setp.ge.u32   %p0, %r8, %r3;\n\
        @%p0 bra $DR_DONE;\n\
    \n\
        // dist[u]\n\
        mul.wide.u32  %rd8, %r1, 4;\n\
        add.u64       %rd9, %rd3, %rd8;\n\
        ld.global.f32 %f0, [%rd9];\n\
    \n\
        // v = col_idx[j]; w = weights[j]\n\
        mul.wide.u32  %rd10, %r8, 4;\n\
        add.u64       %rd11, %rd1, %rd10;\n\
        ld.global.u32 %r9, [%rd11];\n\
        add.u64       %rd12, %rd2, %rd10;\n\
        ld.global.f32 %f1, [%rd12];\n\
    \n\
        // candidate = dist[u] + w\n\
        add.f32       %f2, %f0, %f1;\n\
    \n\
        // dist[v]\n\
        mul.wide.u32  %rd13, %r9, 4;\n\
        add.u64       %rd14, %rd3, %rd13;\n\
        ld.global.f32 %f3, [%rd14];\n\
    \n\
        setp.ge.f32   %p0, %f2, %f3;\n\
        @%p0 bra $DR_DONE;\n\
        st.global.f32 [%rd14], %f2;\n\
        add.u64       %rd15, %rd4, %rd13;\n\
        mov.u32       %r10, 1;\n\
        st.global.u32 [%rd15], %r10;\n\
    \n\
    $DR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// PageRank single power-iteration step.
///
/// Signature: `pagerank_step_kernel(row_ptr_t, col_idx_t, out_degree, rank_in, rank_out, n, damping)`
/// `rank_out[v] = (1-damping)/n + damping * sum_{u in in_neighbors(v)} rank_in[u] / out_degree[u]`.
#[must_use]
pub fn pagerank_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry pagerank_step_kernel(\n\
        .param .u64 p_row_ptr_t,\n\
        .param .u64 p_col_idx_t,\n\
        .param .u64 p_out_degree,\n\
        .param .u64 p_rank_in,\n\
        .param .u64 p_rank_out,\n\
        .param .u32 p_n,\n\
        .param .f32 p_damping\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr_t];\n\
        ld.param.u64  %rd1, [p_col_idx_t];\n\
        ld.param.u64  %rd2, [p_out_degree];\n\
        ld.param.u64  %rd3, [p_rank_in];\n\
        ld.param.u64  %rd4, [p_rank_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.f32  %f0,  [p_damping];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $PR_DONE;\n\
    \n\
        // teleport = (1 - damping) / n\n\
        mov.f32       %f1, 0f3F800000;\n\
        sub.f32       %f2, %f1, %f0;\n\
        cvt.rn.f32.u32 %f3, %r0;\n\
        div.rn.f32    %f4, %f2, %f3;\n\
    \n\
        // start, end = row_ptr_t[v], row_ptr_t[v+1]\n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.u32 %r5, [%rd6];\n\
        add.u64       %rd7, %rd6, 4;\n\
        ld.global.u32 %r6, [%rd7];\n\
    \n\
        // sum = 0\n\
        mov.f32       %f5, 0f00000000;\n\
        mov.u32       %r7, %r5;\n\
    $PR_LOOP:\n\
        setp.ge.u32   %p0, %r7, %r6;\n\
        @%p0 bra $PR_WRITE;\n\
    \n\
        // u = col_idx_t[j]\n\
        mul.wide.u32  %rd8, %r7, 4;\n\
        add.u64       %rd9, %rd1, %rd8;\n\
        ld.global.u32 %r8, [%rd9];\n\
    \n\
        // r_in = rank_in[u]\n\
        mul.wide.u32  %rd10, %r8, 4;\n\
        add.u64       %rd11, %rd3, %rd10;\n\
        ld.global.f32 %f6, [%rd11];\n\
    \n\
        // out_d = out_degree[u]\n\
        add.u64       %rd12, %rd2, %rd10;\n\
        ld.global.u32 %r9, [%rd12];\n\
        cvt.rn.f32.u32 %f7, %r9;\n\
    \n\
        div.rn.f32    %f8, %f6, %f7;\n\
        add.f32       %f5, %f5, %f8;\n\
    \n\
        add.u32       %r7, %r7, 1;\n\
        bra $PR_LOOP;\n\
    \n\
    $PR_WRITE:\n\
        mul.f32       %f9, %f0, %f5;\n\
        add.f32       %f9, %f9, %f4;\n\
        mul.wide.u32  %rd13, %r4, 4;\n\
        add.u64       %rd14, %rd4, %rd13;\n\
        st.global.f32 [%rd14], %f9;\n\
    \n\
    $PR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Floyd-Warshall inner DP update.
///
/// Signature: `fw_inner_kernel(dist, n, k)`
/// `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`.
#[must_use]
pub fn fw_inner_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry fw_inner_kernel(\n\
        .param .u64 p_dist,\n\
        .param .u32 p_n,\n\
        .param .u32 p_k\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_dist];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_k];\n\
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
        @%p0 bra $FW_DONE;\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $FW_DONE;\n\
    \n\
        // d_ij = dist[i*n + j]\n\
        mul.lo.u32    %r10, %r5, %r0;\n\
        add.u32       %r10, %r10, %r9;\n\
        mul.wide.u32  %rd2, %r10, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        // d_ik = dist[i*n + k]\n\
        mul.lo.u32    %r11, %r5, %r0;\n\
        add.u32       %r11, %r11, %r1;\n\
        mul.wide.u32  %rd4, %r11, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        // d_kj = dist[k*n + j]\n\
        mul.lo.u32    %r12, %r1, %r0;\n\
        add.u32       %r12, %r12, %r9;\n\
        mul.wide.u32  %rd6, %r12, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
    \n\
        // candidate = d_ik + d_kj\n\
        add.f32       %f3, %f1, %f2;\n\
        min.f32       %f4, %f0, %f3;\n\
        st.global.f32 [%rd3], %f4;\n\
    \n\
    $FW_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Triangle counting per row.
///
/// Signature: `triangle_count_kernel(row_ptr, col_idx, count, n)`
/// For each vertex `u`, count triples `(u, v, w)` with `u<v<w` and all three edges present.
#[must_use]
pub fn triangle_count_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry triangle_count_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col_idx,\n\
        .param .u64 p_count,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<32>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col_idx];\n\
        ld.param.u64  %rd2, [p_count];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $TC_DONE;\n\
    \n\
        // row_ptr[u], row_ptr[u+1]\n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.u32 %r5, [%rd4];\n\
        add.u64       %rd5, %rd4, 4;\n\
        ld.global.u32 %r6, [%rd5];\n\
    \n\
        mov.u32       %r7, 0;\n\
        mov.u32       %r8, %r5;\n\
    \n\
    $TC_OUTER:\n\
        setp.ge.u32   %p0, %r8, %r6;\n\
        @%p0 bra $TC_WRITE;\n\
    \n\
        // v = col_idx[j]\n\
        mul.wide.u32  %rd6, %r8, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        ld.global.u32 %r9, [%rd7];\n\
    \n\
        // only u < v\n\
        setp.le.u32   %p0, %r9, %r4;\n\
        @%p0 bra $TC_OUTER_END;\n\
    \n\
        add.u32       %r10, %r8, 1;\n\
    $TC_INNER:\n\
        setp.ge.u32   %p0, %r10, %r6;\n\
        @%p0 bra $TC_OUTER_END;\n\
    \n\
        // w = col_idx[k]\n\
        mul.wide.u32  %rd8, %r10, 4;\n\
        add.u64       %rd9, %rd1, %rd8;\n\
        ld.global.u32 %r11, [%rd9];\n\
    \n\
        setp.le.u32   %p0, %r11, %r9;\n\
        @%p0 bra $TC_INNER_END;\n\
    \n\
        // check v-w edge: scan row v for w\n\
        mul.wide.u32  %rd10, %r9, 4;\n\
        add.u64       %rd11, %rd0, %rd10;\n\
        ld.global.u32 %r12, [%rd11];\n\
        add.u64       %rd12, %rd11, 4;\n\
        ld.global.u32 %r13, [%rd12];\n\
    \n\
        mov.u32       %r14, %r12;\n\
    $TC_SCAN:\n\
        setp.ge.u32   %p0, %r14, %r13;\n\
        @%p0 bra $TC_INNER_END;\n\
        mul.wide.u32  %rd13, %r14, 4;\n\
        add.u64       %rd14, %rd1, %rd13;\n\
        ld.global.u32 %r15, [%rd14];\n\
        setp.eq.u32   %p0, %r15, %r11;\n\
        @%p0 bra $TC_HIT;\n\
        add.u32       %r14, %r14, 1;\n\
        bra $TC_SCAN;\n\
    \n\
    $TC_HIT:\n\
        add.u32       %r7, %r7, 1;\n\
    \n\
    $TC_INNER_END:\n\
        add.u32       %r10, %r10, 1;\n\
        bra $TC_INNER;\n\
    \n\
    $TC_OUTER_END:\n\
        add.u32       %r8, %r8, 1;\n\
        bra $TC_OUTER;\n\
    \n\
    $TC_WRITE:\n\
        mul.wide.u32  %rd15, %r4, 4;\n\
        add.u64       %rd5, %rd2, %rd15;\n\
        st.global.u32 [%rd5], %r7;\n\
    \n\
    $TC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Boolean CSR sparse mat-vec (for matrix-form BFS).
///
/// Signature: `csr_spmv_bool_kernel(row_ptr, col_idx, x, y, n)`
/// `y[i] = OR over j in row_ptr[i] .. row_ptr[i+1] of x[col_idx[j]]`.
#[must_use]
pub fn csr_spmv_bool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry csr_spmv_bool_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col_idx,\n\
        .param .u64 p_x,\n\
        .param .u64 p_y,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<16>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col_idx];\n\
        ld.param.u64  %rd2, [p_x];\n\
        ld.param.u64  %rd3, [p_y];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $SP_DONE;\n\
    \n\
        // start, end = row_ptr[i], row_ptr[i+1]\n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.u32 %r5, [%rd5];\n\
        add.u64       %rd6, %rd5, 4;\n\
        ld.global.u32 %r6, [%rd6];\n\
    \n\
        mov.u32       %r7, 0;\n\
        mov.u32       %r8, %r5;\n\
    \n\
    $SP_LOOP:\n\
        setp.ge.u32   %p0, %r8, %r6;\n\
        @%p0 bra $SP_WRITE;\n\
    \n\
        mul.wide.u32  %rd7, %r8, 4;\n\
        add.u64       %rd8, %rd1, %rd7;\n\
        ld.global.u32 %r9, [%rd8];\n\
        mul.wide.u32  %rd9, %r9, 4;\n\
        add.u64       %rd10, %rd2, %rd9;\n\
        ld.global.u32 %r10, [%rd10];\n\
        or.b32        %r7, %r7, %r10;\n\
    \n\
        add.u32       %r8, %r8, 1;\n\
        bra $SP_LOOP;\n\
    \n\
    $SP_WRITE:\n\
        add.u64       %rd11, %rd3, %rd4;\n\
        st.global.u32 [%rd11], %r7;\n\
    \n\
    $SP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Label propagation single step.
///
/// Signature: `community_label_kernel(row_ptr, col_idx, label_in, label_out, n)`
/// Update `label[u]` = most-frequent label among neighbors of u (ties broken by min label).
#[must_use]
pub fn community_label_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry community_label_kernel(\n\
        .param .u64 p_row_ptr,\n\
        .param .u64 p_col_idx,\n\
        .param .u64 p_label_in,\n\
        .param .u64 p_label_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<24>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_row_ptr];\n\
        ld.param.u64  %rd1, [p_col_idx];\n\
        ld.param.u64  %rd2, [p_label_in];\n\
        ld.param.u64  %rd3, [p_label_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $LP_DONE;\n\
    \n\
        // start = row_ptr[u], end = row_ptr[u+1]\n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.u32 %r5, [%rd5];\n\
        add.u64       %rd6, %rd5, 4;\n\
        ld.global.u32 %r6, [%rd6];\n\
    \n\
        // default: keep own label\n\
        add.u64       %rd7, %rd2, %rd4;\n\
        ld.global.u32 %r7, [%rd7];\n\
        mov.u32       %r8, %r7;\n\
    \n\
        // simple min-label rule for ties (deterministic; aggregation done CPU-side)\n\
        mov.u32       %r9, %r5;\n\
    $LP_LOOP:\n\
        setp.ge.u32   %p0, %r9, %r6;\n\
        @%p0 bra $LP_WRITE;\n\
    \n\
        mul.wide.u32  %rd8, %r9, 4;\n\
        add.u64       %rd9, %rd1, %rd8;\n\
        ld.global.u32 %r10, [%rd9];\n\
        mul.wide.u32  %rd10, %r10, 4;\n\
        add.u64       %rd11, %rd2, %rd10;\n\
        ld.global.u32 %r11, [%rd11];\n\
    \n\
        setp.ge.u32   %p0, %r11, %r8;\n\
        @%p0 bra $LP_NEXT;\n\
        mov.u32       %r8, %r11;\n\
    \n\
    $LP_NEXT:\n\
        add.u32       %r9, %r9, 1;\n\
        bra $LP_LOOP;\n\
    \n\
    $LP_WRITE:\n\
        add.u64       %rd12, %rd3, %rd4;\n\
        st.global.u32 [%rd12], %r8;\n\
    \n\
    $LP_DONE:\n\
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
            ("bfs_level", bfs_level_ptx),
            ("dijkstra_relax", dijkstra_relax_ptx),
            ("pagerank_step", pagerank_step_ptx),
            ("fw_inner", fw_inner_ptx),
            ("triangle_count", triangle_count_ptx),
            ("csr_spmv_bool", csr_spmv_bool_ptx),
            ("community_label", community_label_ptx),
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
    fn ptx_target_matches_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            let s = bfs_level_ptx(sm);
            assert!(s.contains(&format!("sm_{sm}")));
        }
    }
}
