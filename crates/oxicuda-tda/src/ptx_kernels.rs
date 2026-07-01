//! GPU PTX kernels for Topological Data Analysis algorithms.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM≥100 → 8.7 (Blackwell), SM≥90 → 8.4 (Hopper),
//!     SM≥80  → 8.0 (Ampere),    else → 7.5 (Turing).
//!
//! IMPORTANT: PTX kernel bodies use **string concatenation** (NOT `format!()`) for
//! sections containing `%rd`, `%r`, `%f` register names, which Rust's format macro
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

/// Encode a `f32` constant as a PTX immediate hex literal (`0Fxxxxxxxx`).
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

/// Tiled pairwise squared Euclidean distance kernel.
///
/// Signature: `pairwise_dist_kernel(points: *f32, dist: *f32, n_points: u32, n_dims: u32)`
/// Grid = (ceil(n/16), ceil(n/16), 1), Block = (16, 16, 1).
/// Each thread computes `dist[i, j] = sum_d (points[i,d] - points[j,d])^2`.
#[must_use]
pub fn pairwise_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry pairwise_dist_kernel(\n\
        .param .u64 p_points,\n\
        .param .u64 p_dist,\n\
        .param .u32 p_n_points,\n\
        .param .u32 p_n_dims\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_points];\n\
        ld.param.u64  %rd1, [p_dist];\n\
        ld.param.u32  %r0,  [p_n_points];\n\
        ld.param.u32  %r1,  [p_n_dims];\n\
    \n\
        // row i = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r2, %ntid.y;\n\
        mov.u32       %r3, %ctaid.y;\n\
        mov.u32       %r4, %tid.y;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        // col j = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r6, %ntid.x;\n\
        mov.u32       %r7, %ctaid.x;\n\
        mov.u32       %r8, %tid.x;\n\
        mad.lo.u32    %r9, %r6, %r7, %r8;\n\
    \n\
        // guard: i < n_points && j < n_points\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $PD_DONE;\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $PD_DONE;\n\
    \n\
        // accumulate squared distance\n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r10, 0;\n\
    \n\
    $PD_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r1;\n\
        @%p0 bra $PD_WRITE;\n\
    \n\
        // points[i * n_dims + d]\n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        // points[j * n_dims + d]\n\
        mul.lo.u32    %r12, %r9, %r1;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd2, %r12, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd3];\n\
    \n\
        sub.f32       %f3, %f1, %f2;\n\
        fma.rn.f32    %f0, %f3, %f3, %f0;\n\
    \n\
        add.u32       %r10, %r10, 1;\n\
        bra $PD_LOOP;\n\
    \n\
    $PD_WRITE:\n\
        // dist[i * n_points + j] = f0\n\
        mul.lo.u32    %r13, %r5, %r0;\n\
        add.u32       %r13, %r13, %r9;\n\
        mul.wide.u32  %rd4, %r13, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        st.global.f32 [%rd5], %f0;\n\
    \n\
    $PD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Radix-sort pass: sort simplex indices by filtration value.
///
/// Signature: `filtration_sort_kernel(filt_values: *f32, indices: *u32, n: u32)`
/// Single-pass partial sort: each thread writes its index sorted by float key.
#[must_use]
pub fn filtration_sort_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry filtration_sort_kernel(\n\
        .param .u64 p_filt_values,\n\
        .param .u64 p_indices,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_filt_values];\n\
        ld.param.u64  %rd1, [p_indices];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        // tid = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $FS_DONE;\n\
    \n\
        // load my filtration value\n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        // Count how many values are strictly smaller\n\
        mov.u32       %r5, 0;\n\
        mov.u32       %r6, 0;\n\
    \n\
    $FS_LOOP:\n\
        setp.ge.u32   %p0, %r6, %r0;\n\
        @%p0 bra $FS_WRITE;\n\
    \n\
        mul.wide.u32  %rd4, %r6, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        setp.lt.f32   %p0, %f1, %f0;\n\
        @%p0 add.u32  %r5, %r5, 1;\n\
    \n\
        add.u32       %r6, %r6, 1;\n\
        bra $FS_LOOP;\n\
    \n\
    $FS_WRITE:\n\
        // indices[r5] = r4 (approximate radix sort, collision-free for distinct values)\n\
        mul.wide.u32  %rd6, %r5, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.u32 [%rd7], %r4;\n\
    \n\
    $FS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Parallel low-column identification for boundary matrix reduction.
///
/// Signature: `boundary_reduce_kernel(pivot_col: *i32, n_cols: u32)`
/// Each thread scans column j to find its lowest nonzero row.
#[must_use]
pub fn boundary_reduce_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry boundary_reduce_kernel(\n\
        .param .u64 p_pivot_col,\n\
        .param .u32 p_n_cols\n\
    )\n\
    {\n\
        .reg .u64  %rd<6>;\n\
        .reg .u32  %r<10>;\n\
        .reg .s32  %sr<2>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_pivot_col];\n\
        ld.param.u32  %r0,  [p_n_cols];\n\
    \n\
        // tid = global thread index\n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $BR_DONE;\n\
    \n\
        // load pivot_col[tid]: -1 means zero column, else the pivot row\n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.s32 %sr0, [%rd3];\n\
    \n\
        // Atomically mark: if pivot_col[tid] >= 0, write tid into pivot_col[pivot_row]\n\
        mov.s32       %sr1, -1;\n\
        setp.eq.s32   %p0, %sr0, %sr1;\n\
        @%p0 bra $BR_DONE;\n\
    \n\
        // Convert pivot row to address and write column index.\n\
        // NOTE: atom.exch only supports bit types (.b32/.b64), never .s32; the\n\
        // stored value is tid (%r4, a non-negative column index) and the old\n\
        // value goes to the discarded %r6.\n\
        cvt.u32.s32   %r5, %sr0;\n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        atom.global.exch.b32 %r6, [%rd5], %r4;\n\
    \n\
    $BR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Wasserstein cost matrix between two persistence diagrams.
///
/// Signature: `diagram_match_kernel(birth_a, death_a, birth_b, death_b, cost, n_a, n_b)`
/// `cost[i, j] = max(|birth_a[i] - birth_b[j]|, |death_a[i] - death_b[j]|)`
#[must_use]
pub fn diagram_match_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry diagram_match_kernel(\n\
        .param .u64 p_birth_a,\n\
        .param .u64 p_death_a,\n\
        .param .u64 p_birth_b,\n\
        .param .u64 p_death_b,\n\
        .param .u64 p_cost,\n\
        .param .u32 p_n_a,\n\
        .param .u32 p_n_b\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_birth_a];\n\
        ld.param.u64  %rd1, [p_death_a];\n\
        ld.param.u64  %rd2, [p_birth_b];\n\
        ld.param.u64  %rd3, [p_death_b];\n\
        ld.param.u64  %rd4, [p_cost];\n\
        ld.param.u32  %r0,  [p_n_a];\n\
        ld.param.u32  %r1,  [p_n_b];\n\
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
        @%p0 bra $DM_DONE;\n\
        setp.ge.u32   %p0, %r9, %r1;\n\
        @%p0 bra $DM_DONE;\n\
    \n\
        // load birth_a[i], death_a[i]\n\
        mul.wide.u32  %rd5, %r5, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f0, [%rd6];\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
    \n\
        // load birth_b[j], death_b[j]\n\
        mul.wide.u32  %rd7, %r9, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        ld.global.f32 %f2, [%rd8];\n\
        add.u64       %rd8, %rd3, %rd7;\n\
        ld.global.f32 %f3, [%rd8];\n\
    \n\
        // cost = max(|ba-bb|, |da-db|)\n\
        sub.f32       %f4, %f0, %f2;\n\
        abs.f32       %f4, %f4;\n\
        sub.f32       %f5, %f1, %f3;\n\
        abs.f32       %f5, %f5;\n\
        max.f32       %f6, %f4, %f5;\n\
    \n\
        // store cost[i * n_b + j]\n\
        mul.lo.u32    %r10, %r5, %r1;\n\
        add.u32       %r10, %r10, %r9;\n\
        mul.wide.u32  %rd9, %r10, 4;\n\
        add.u64       %rd9, %rd4, %rd9;\n\
        st.global.f32 [%rd9], %f6;\n\
    \n\
    $DM_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Landmark-to-point distances for witness complex construction.
///
/// Signature: `witness_dist_kernel(points, landmarks, dist, n_pts, n_land, n_dims)`
/// `dist[l * n_pts + w]` = euclidean(`landmarks[l]`, `points[w]`)
#[must_use]
pub fn witness_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry witness_dist_kernel(\n\
        .param .u64 p_points,\n\
        .param .u64 p_landmarks,\n\
        .param .u64 p_dist,\n\
        .param .u32 p_n_pts,\n\
        .param .u32 p_n_land,\n\
        .param .u32 p_n_dims\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<14>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_points];\n\
        ld.param.u64  %rd1, [p_landmarks];\n\
        ld.param.u64  %rd2, [p_dist];\n\
        ld.param.u32  %r0,  [p_n_pts];\n\
        ld.param.u32  %r1,  [p_n_land];\n\
        ld.param.u32  %r2,  [p_n_dims];\n\
    \n\
        // landmark l = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r3, %ntid.y;\n\
        mov.u32       %r4, %ctaid.y;\n\
        mov.u32       %r5, %tid.y;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
    \n\
        // witness w = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r7, %ntid.x;\n\
        mov.u32       %r8, %ctaid.x;\n\
        mov.u32       %r9, %tid.x;\n\
        mad.lo.u32    %r10, %r7, %r8, %r9;\n\
    \n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $WD_DONE;\n\
        setp.ge.u32   %p0, %r10, %r0;\n\
        @%p0 bra $WD_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r11, 0;\n\
    \n\
    $WD_LOOP:\n\
        setp.ge.u32   %p0, %r11, %r2;\n\
        @%p0 bra $WD_WRITE;\n\
    \n\
        // landmarks[l * n_dims + d]\n\
        mul.lo.u32    %r12, %r6, %r2;\n\
        add.u32       %r12, %r12, %r11;\n\
        mul.wide.u32  %rd3, %r12, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // points[w * n_dims + d]\n\
        mul.lo.u32    %r13, %r10, %r2;\n\
        add.u32       %r13, %r13, %r11;\n\
        mul.wide.u32  %rd5, %r13, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        sub.f32       %f3, %f1, %f2;\n\
        fma.rn.f32    %f0, %f3, %f3, %f0;\n\
    \n\
        add.u32       %r11, %r11, 1;\n\
        bra $WD_LOOP;\n\
    \n\
    $WD_WRITE:\n\
        sqrt.rn.f32   %f4, %f0;\n\
        // dist[l * n_pts + w]\n\
        mul.lo.u32    %r12, %r6, %r0;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd7, %r12, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        st.global.f32 [%rd8], %f4;\n\
    \n\
    $WD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Count birth-death pairs per dimension in persistence pairs array.
///
/// Signature: `betti_count_kernel(dims, deaths, betti, n_pairs, query_dim, max_death)`
/// Increments `betti[0]` for each pair with dim == query_dim and death > max_death (essential).
#[must_use]
pub fn betti_count_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let max_death_hex = f32_hex(f32::INFINITY);
    let body = ".visible .entry betti_count_kernel(\n\
        .param .u64 p_dims,\n\
        .param .u64 p_deaths,\n\
        .param .u64 p_betti,\n\
        .param .u32 p_n_pairs,\n\
        .param .u32 p_query_dim,\n\
        .param .f32 p_max_death\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .s32  %sr0;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_dims];\n\
        ld.param.u64  %rd1, [p_deaths];\n\
        ld.param.u64  %rd2, [p_betti];\n\
        ld.param.u32  %r0,  [p_n_pairs];\n\
        ld.param.u32  %r1,  [p_query_dim];\n\
        ld.param.f32  %f0,  [p_max_death];\n\
    \n\
        // tid = global thread index\n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $BC_DONE;\n\
    \n\
        // load dims[tid]\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.s32 %sr0, [%rd4];\n\
        cvt.u32.s32   %r6, %sr0;\n\
    \n\
        // check dim == query_dim\n\
        setp.ne.u32   %p0, %r6, %r1;\n\
        @%p0 bra $BC_DONE;\n\
    \n\
        // load deaths[tid]\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        // check death > max_death (i.e., essential: death == infinity)\n";
    let body2 = "        mov.f32       %f2, ";
    let body3 = ";\n\
        setp.eq.f32   %p0, %f1, %f2;\n\
        @%p0 atom.global.add.u32 %r7, [%rd2], 1;\n\
    \n\
    $BC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body + body2 + &max_death_hex + body3
}

/// Single-linkage clustering distances for Mapper algorithm.
///
/// Signature: `mapper_cluster_kernel(points, cluster_id, n_pts, n_dims, threshold)`
/// Computes pairwise distances and marks pairs within threshold for union-find.
#[must_use]
pub fn mapper_cluster_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mapper_cluster_kernel(\n\
        .param .u64 p_points,\n\
        .param .u64 p_cluster_id,\n\
        .param .u32 p_n_pts,\n\
        .param .u32 p_n_dims,\n\
        .param .f32 p_threshold\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<14>;\n\
        .reg .s32  %sr<4>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_points];\n\
        ld.param.u64  %rd1, [p_cluster_id];\n\
        ld.param.u32  %r0,  [p_n_pts];\n\
        ld.param.u32  %r1,  [p_n_dims];\n\
        ld.param.f32  %f0,  [p_threshold];\n\
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
        @%p0 bra $MC_DONE;\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $MC_DONE;\n\
    \n\
        // Only process upper triangle i < j\n\
        setp.ge.u32   %p0, %r5, %r9;\n\
        @%p0 bra $MC_DONE;\n\
    \n\
        // Compute squared distance between points[i] and points[j]\n\
        mov.f32       %f1, 0f00000000;\n\
        mov.u32       %r10, 0;\n\
    \n\
    $MC_LOOP:\n\
        setp.ge.u32   %p0, %r10, %r1;\n\
        @%p0 bra $MC_CHECK;\n\
    \n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd3];\n\
    \n\
        mul.lo.u32    %r12, %r9, %r1;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd4, %r12, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f3, [%rd5];\n\
    \n\
        sub.f32       %f4, %f2, %f3;\n\
        fma.rn.f32    %f1, %f4, %f4, %f1;\n\
    \n\
        add.u32       %r10, %r10, 1;\n\
        bra $MC_LOOP;\n\
    \n\
    $MC_CHECK:\n\
        sqrt.rn.f32   %f5, %f1;\n\
        setp.gt.f32   %p0, %f5, %f0;\n\
        @%p0 bra $MC_DONE;\n\
    \n\
        // Mark cluster_id[j] = cluster_id[i] (single-linkage step)\n\
        mul.wide.u32  %rd6, %r5, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        ld.global.s32 %sr0, [%rd7];\n\
        mul.wide.u32  %rd8, %r9, 4;\n\
        add.u64       %rd9, %rd1, %rd8;\n\
        st.global.s32 [%rd9], %sr0;\n\
    \n\
    $MC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}
