//! PTX GPU kernel sources for GNN operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`csr_spmv_ptx`] | Sparse matrix-vector multiply (warp-per-row) |
//! | [`scatter_add_ptx`] | Scatter-add for message aggregation |
//! | [`gat_attention_ptx`] | GAT edge attention score computation |
//! | [`softmax_edge_ptx`] | Per-source edge softmax normalisation |
//! | [`aggregate_mean_ptx`] | Degree-normalised neighbourhood mean aggregation |
//! | [`gin_combine_ptx`] | GIN (1+ε) self + aggregated feature combine |
//! | [`topk_score_ptx`] | TopK projection scoring with tanh |

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let ptx_ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ptx_ver}\n.target sm_{sm}\n.address_size 64\n\n")
}

/// Encode an `f32` as its IEEE-754 hex literal for PTX `mov.f32`.
///
/// Example: `f32_hex(0.2f32)` → `"0F3E4CCCCD"` (PTX hex float notation).
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: csr_spmv ──────────────────────────────────────────────────────

/// CSR sparse matrix–vector multiply: `y[i] = Σ_{j∈row(i)} val[e] * x[col[e]]`.
///
/// Uses a warp-per-row strategy:
/// - `tid  = blockIdx * blockDim + threadIdx`
/// - `row  = tid / 32`  (one warp handles one row)
/// - `lane = tid % 32`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_row_ptr` | `u64` → `u32*` | Row pointer array `[n_rows+1]` |
/// | `p_col_idx` | `u64` → `u32*` | Column index array `[n_edges]` |
/// | `p_val`     | `u64` → `f32*` | Non-zero value array `[n_edges]` |
/// | `p_x`       | `u64` → `f32*` | Input vector `[n_rows]` |
/// | `p_y`       | `u64` → `f32*` | Output vector `[n_rows]` |
/// | `n_rows`    | `u32`          | Number of rows |
///
/// Launch with `grid = ceil(n_rows * 32 / 256)`, `block = 256`.
pub fn csr_spmv_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry csr_spmv(
    .param .u64 p_row_ptr,
    .param .u64 p_col_idx,
    .param .u64 p_val,
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u32 n_rows
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p<4>;

    // Load parameters
    ld.param.u64  %rd0, [p_row_ptr];
    ld.param.u64  %rd1, [p_col_idx];
    ld.param.u64  %rd2, [p_val];
    ld.param.u64  %rd3, [p_x];
    ld.param.u64  %rd4, [p_y];
    ld.param.u32  %r0,  [n_rows];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;   // r4 = tid

    // row = tid / 32,  lane = tid % 32
    shr.u32       %r5, %r4, 5;           // r5 = row = tid >> 5
    and.b32       %r6, %r4, 31;          // r6 = lane = tid & 31

    // if row >= n_rows, exit
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $DONE;

    // row_start = row_ptr[row],  row_end = row_ptr[row+1]
    mul.wide.u32  %rd5, %r5, 4;
    add.u64       %rd5, %rd0, %rd5;
    ld.global.u32 %r7, [%rd5];           // r7 = row_start

    add.u32       %r8, %r5, 1;
    mul.wide.u32  %rd6, %r8, 4;
    add.u64       %rd6, %rd0, %rd6;
    ld.global.u32 %r8, [%rd6];           // r8 = row_end

    // Warp iterates: e = row_start + lane, step = 32
    add.u32       %r9, %r7, %r6;         // r9 = e = row_start + lane
    mov.f32       %f0, 0F00000000;        // f0 = partial_sum = 0.0

$LOOP:
    setp.ge.u32   %p1, %r9, %r8;
    @%p1 bra $REDUCE;

    // col = col_idx[e]
    mul.wide.u32  %rd7, %r9, 4;
    add.u64       %rd7, %rd1, %rd7;
    ld.global.u32 %r10, [%rd7];          // r10 = col

    // val = p_val[e]
    mul.wide.u32  %rd8, %r9, 4;
    add.u64       %rd8, %rd2, %rd8;
    ld.global.f32 %f1, [%rd8];           // f1 = val

    // x_col = p_x[col]
    mul.wide.u32  %rd9, %r10, 4;
    add.u64       %rd9, %rd3, %rd9;
    ld.global.f32 %f2, [%rd9];           // f2 = x[col]

    fma.rn.f32    %f0, %f1, %f2, %f0;   // f0 += val * x[col]

    add.u32       %r9, %r9, 32;          // e += warp_size
    bra           $LOOP;

$REDUCE:
    // Warp-level reduction using shfl_down (butterfly pattern across 16-8-4-2-1 lanes)
    shfl.sync.down.b32 %f3, %f0, 16, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 8, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 4, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 2, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;
    shfl.sync.down.b32 %f3, %f0, 1, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;

    // Only lane 0 writes result
    setp.ne.u32   %p2, %r6, 0;
    @%p2 bra $DONE;

    mul.wide.u32  %rd10, %r5, 4;
    add.u64       %rd10, %rd4, %rd10;
    st.global.f32 [%rd10], %f0;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: scatter_add ───────────────────────────────────────────────────

/// Scatter-add: `out[idx[i]] += src[i]` for i in [0, n_edges).
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_idx` | `u64` → `u32*` | Destination index array `[n]` |
/// | `p_src` | `u64` → `f32*` | Source value array `[n]` |
/// | `p_out` | `u64` → `f32*` | Output accumulator array |
/// | `n`     | `u32`          | Number of elements |
///
/// Uses `atom.global.add.f32`. Launch with `grid = ceil(n / 256)`, `block = 256`.
pub fn scatter_add_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry scatter_add(
    .param .u64 p_idx,
    .param .u64 p_src,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<8>;
    .reg .f32  %f<2>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_idx];
    ld.param.u64  %rd1, [p_src];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    // dest_idx = idx[tid]
    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.u32 %r5, [%rd3];

    // val = src[tid]
    mul.wide.u32  %rd4, %r4, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.f32 %f0, [%rd4];

    // atom.add out[dest_idx] += val
    mul.wide.u32  %rd5, %r5, 4;
    add.u64       %rd5, %rd2, %rd5;
    atom.global.add.f32 %f1, [%rd5], %f0;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: gat_attention ─────────────────────────────────────────────────

/// GAT edge attention score computation.
///
/// `score[e] = LeakyReLU(a^T [Wx_src || Wx_dst])` where `||` is concatenation
/// and LeakyReLU uses slope 0.2.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_src_feat` | `u64` → `f32*` | Projected source features `[n_edges × feat_dim]` |
/// | `p_dst_feat` | `u64` → `f32*` | Projected destination features `[n_edges × feat_dim]` |
/// | `p_a`        | `u64` → `f32*` | Attention weight vector `[2 × feat_dim]` |
/// | `p_score`    | `u64` → `f32*` | Output attention scores `[n_edges]` |
/// | `feat_dim`   | `u32`          | Feature dimension per head |
/// | `n_edges`    | `u32`          | Number of edges |
///
/// One thread per edge. Launch with `grid = ceil(n_edges / 256)`, `block = 256`.
pub fn gat_attention_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let leaky_slope = f32_hex(0.2_f32);
    format!(
        r#"{hdr}.visible .entry gat_attention(
    .param .u64 p_src_feat,
    .param .u64 p_dst_feat,
    .param .u64 p_a,
    .param .u64 p_score,
    .param .u32 feat_dim,
    .param .u32 n_edges
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<12>;
    .reg .f32  %f<8>;
    .reg .pred %p<4>;

    ld.param.u64  %rd0, [p_src_feat];
    ld.param.u64  %rd1, [p_dst_feat];
    ld.param.u64  %rd2, [p_a];
    ld.param.u64  %rd3, [p_score];
    ld.param.u32  %r0,  [feat_dim];
    ld.param.u32  %r1,  [n_edges];

    // tid
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // r5 = edge_id

    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $DONE;

    // base offset for this edge in feat arrays: edge_id * feat_dim * 4
    mul.lo.u32    %r6, %r5, %r0;         // r6 = edge_id * feat_dim
    mul.wide.u32  %rd4, %r6, 4;          // rd4 = byte offset

    // dot product: sum_k src[edge*fd + k] * a[k]  +  dst[edge*fd + k] * a[fd + k]
    mov.f32       %f0, 0F00000000;        // accumulator = 0
    mov.u32       %r7, 0;                 // k = 0

$LOOP:
    setp.ge.u32   %p1, %r7, %r0;
    @%p1 bra $POSTLOOP;

    // byte offset for element k within this edge's feature slice
    mul.wide.u32  %rd5, %r7, 4;

    // src_feat[edge*fd + k]
    add.u64       %rd6, %rd0, %rd4;
    add.u64       %rd6, %rd6, %rd5;
    ld.global.f32 %f1, [%rd6];

    // a[k]
    mul.wide.u32  %rd7, %r7, 4;
    add.u64       %rd7, %rd2, %rd7;
    ld.global.f32 %f2, [%rd7];

    fma.rn.f32    %f0, %f1, %f2, %f0;   // accum += src[k] * a[k]

    // dst_feat[edge*fd + k]
    add.u64       %rd8, %rd1, %rd4;
    add.u64       %rd8, %rd8, %rd5;
    ld.global.f32 %f3, [%rd8];

    // a[fd + k]
    add.u32       %r8, %r7, %r0;         // fd + k
    mul.wide.u32  %rd9, %r8, 4;
    add.u64       %rd9, %rd2, %rd9;
    ld.global.f32 %f4, [%rd9];

    fma.rn.f32    %f0, %f3, %f4, %f0;   // accum += dst[k] * a[fd+k]

    add.u32       %r7, %r7, 1;
    bra           $LOOP;

$POSTLOOP:
    // LeakyReLU: if x < 0 then 0.2 * x else x
    mov.f32       %f5, {leaky_slope};
    setp.lt.f32   %p2, %f0, 0F00000000;
    mul.f32       %f6, %f0, %f5;
    selp.f32      %f0, %f6, %f0, %p2;

    // store score[edge_id]
    mul.wide.u32  %rd10, %r5, 4;
    add.u64       %rd10, %rd3, %rd10;
    st.global.f32 [%rd10], %f0;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 4: softmax_edge ──────────────────────────────────────────────────

/// Per-source edge softmax normalisation.
///
/// Numerically stable: `α[e] = exp(score[e] - max_src) / Σ exp(score[e'] - max_src)`
/// where the sum/max are over all edges `e'` sharing the same source node.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_score`   | `u64` → `f32*` | Raw attention scores `[n_edges]` |
/// | `p_row_ptr` | `u64` → `u32*` | Row pointer (out-edges per node) `[n_nodes+1]` |
/// | `p_alpha`   | `u64` → `f32*` | Output normalised weights `[n_edges]` |
/// | `n_nodes`   | `u32`          | Number of source nodes |
///
/// One thread per node. Launch with `grid = ceil(n_nodes / 256)`, `block = 256`.
pub fn softmax_edge_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // log2(e): the SFU only provides `ex2.approx` (base-2), so `exp(x)` must be
    // computed as `2^(x * log2(e))`. Omitting this factor would silently yield a
    // base-2 softmax (a different, hotter distribution), which is why the scale
    // is applied before every `ex2.approx.f32` below — mirroring `topk_score`.
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    format!(
        r#"{hdr}.visible .entry softmax_edge(
    .param .u64 p_score,
    .param .u64 p_row_ptr,
    .param .u64 p_alpha,
    .param .u32 n_nodes
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<8>;
    .reg .pred %p<4>;

    ld.param.u64  %rd0, [p_score];
    ld.param.u64  %rd1, [p_row_ptr];
    ld.param.u64  %rd2, [p_alpha];
    ld.param.u32  %r0,  [n_nodes];

    // log2(e) constant for the base-2 → base-e exp conversion.
    mov.f32       %f5, {log2e};

    // tid = node_id
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $DONE;

    // row_start = row_ptr[node_id]
    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd3, %rd1, %rd3;
    ld.global.u32 %r5, [%rd3];           // r5 = row_start

    // row_end = row_ptr[node_id + 1]
    add.u32       %r6, %r4, 1;
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.u32 %r6, [%rd4];           // r6 = row_end

    // If no outgoing edges, skip
    setp.ge.u32   %p1, %r5, %r6;
    @%p1 bra $DONE;

    // Pass 1: find max score in this node's out-edges
    mov.f32       %f0, 0FFF800000;        // -inf
    mov.u32       %r7, %r5;              // e = row_start

$MAXLOOP:
    setp.ge.u32   %p2, %r7, %r6;
    @%p2 bra $EXPLOOP_INIT;

    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd5, %rd0, %rd5;
    ld.global.f32 %f1, [%rd5];
    max.f32       %f0, %f0, %f1;
    add.u32       %r7, %r7, 1;
    bra           $MAXLOOP;

$EXPLOOP_INIT:
    // Pass 2: compute sum of exp(score - max)
    mov.f32       %f2, 0F00000000;        // sum = 0
    mov.u32       %r7, %r5;

$SUMLOOP:
    setp.ge.u32   %p2, %r7, %r6;
    @%p2 bra $NORMLOOP_INIT;

    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd5, %rd0, %rd5;
    ld.global.f32 %f3, [%rd5];
    sub.f32       %f3, %f3, %f0;         // score - max
    mul.f32       %f3, %f3, %f5;         // (score - max) * log2(e)
    ex2.approx.f32 %f3, %f3;             // exp(score - max) = 2^((score-max)*log2e)
    add.f32       %f2, %f2, %f3;
    add.u32       %r7, %r7, 1;
    bra           $SUMLOOP;

$NORMLOOP_INIT:
    // Pass 3: normalise and store
    mov.u32       %r7, %r5;

$NORMLOOP:
    setp.ge.u32   %p2, %r7, %r6;
    @%p2 bra $DONE;

    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd5, %rd0, %rd5;
    ld.global.f32 %f4, [%rd5];
    sub.f32       %f4, %f4, %f0;
    mul.f32       %f4, %f4, %f5;         // (score - max) * log2(e)
    ex2.approx.f32 %f4, %f4;             // exp(score - max)
    div.approx.f32 %f4, %f4, %f2;

    mul.wide.u32  %rd6, %r7, 4;
    add.u64       %rd6, %rd2, %rd6;
    st.global.f32 [%rd6], %f4;

    add.u32       %r7, %r7, 1;
    bra           $NORMLOOP;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: aggregate_mean ────────────────────────────────────────────────

/// Degree-normalised neighbourhood mean aggregation.
///
/// `out[i*fd + k] = (1/degree[i]) * Σ_{j∈N(i)} feat[j*fd + k]`
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_feat`    | `u64` → `f32*` | Node features `[n_nodes × feat_dim]` |
/// | `p_row_ptr` | `u64` → `u32*` | CSR row pointer `[n_nodes+1]` |
/// | `p_col_idx` | `u64` → `u32*` | CSR column indices `[n_edges]` |
/// | `p_out`     | `u64` → `f32*` | Output aggregated features `[n_nodes × feat_dim]` |
/// | `feat_dim`  | `u32`          | Feature dimension |
/// | `n_nodes`   | `u32`          | Number of nodes |
///
/// Launch with `grid = ceil(n_nodes * feat_dim / 256)`, `block = 256`.
pub fn aggregate_mean_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry aggregate_mean(
    .param .u64 p_feat,
    .param .u64 p_row_ptr,
    .param .u64 p_col_idx,
    .param .u64 p_out,
    .param .u32 feat_dim,
    .param .u32 n_nodes
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<14>;
    .reg .f32  %f<4>;
    .reg .pred %p<4>;

    ld.param.u64  %rd0, [p_feat];
    ld.param.u64  %rd1, [p_row_ptr];
    ld.param.u64  %rd2, [p_col_idx];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [feat_dim];
    ld.param.u32  %r1,  [n_nodes];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    // Each thread handles one (node, feature_dim) pair
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // tid

    // node_id = tid / feat_dim, k = tid % feat_dim
    div.u32       %r6, %r5, %r0;         // node_id
    rem.u32       %r7, %r5, %r0;         // k

    // guard: node_id < n_nodes
    mul.lo.u32    %r8, %r1, %r0;         // n_nodes * feat_dim
    setp.ge.u32   %p0, %r5, %r8;
    @%p0 bra $DONE;

    // row_start = row_ptr[node_id]
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.u32 %r9, [%rd4];

    // row_end = row_ptr[node_id + 1]
    add.u32       %r10, %r6, 1;
    mul.wide.u32  %rd5, %r10, 4;
    add.u64       %rd5, %rd1, %rd5;
    ld.global.u32 %r10, [%rd5];

    // degree = row_end - row_start
    sub.u32       %r11, %r10, %r9;

    // If isolated node, write 0 and exit
    mov.f32       %f0, 0F00000000;
    setp.eq.u32   %p1, %r11, 0;
    @%p1 bra $WRITE;

    // Accumulate sum over neighbours
    mov.u32       %r12, %r9;             // e = row_start

$LOOP:
    setp.ge.u32   %p2, %r12, %r10;
    @%p2 bra $NORMALIZE;

    // neighbour = col_idx[e]
    mul.wide.u32  %rd6, %r12, 4;
    add.u64       %rd6, %rd2, %rd6;
    ld.global.u32 %r13, [%rd6];

    // feat[neighbour*fd + k]
    mad.lo.u32    %r13, %r13, %r0, %r7;
    mul.wide.u32  %rd7, %r13, 4;
    add.u64       %rd7, %rd0, %rd7;
    ld.global.f32 %f1, [%rd7];

    add.f32       %f0, %f0, %f1;
    add.u32       %r12, %r12, 1;
    bra           $LOOP;

$NORMALIZE:
    // out[node*fd + k] = sum / degree
    cvt.rn.f32.u32 %f2, %r11;
    div.approx.f32  %f0, %f0, %f2;

$WRITE:
    mul.wide.u32  %rd8, %r5, 4;
    add.u64       %rd8, %rd3, %rd8;
    st.global.f32 [%rd8], %f0;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 6: gin_combine ───────────────────────────────────────────────────

/// GIN feature combination: `out[i*d + k] = (1+ε)*self_feat[i*d + k] + aggr_feat[i*d + k]`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_self`   | `u64` → `f32*` | Self features `[n × feat_dim]` |
/// | `p_aggr`   | `u64` → `f32*` | Aggregated neighbour features `[n × feat_dim]` |
/// | `p_out`    | `u64` → `f32*` | Output `[n × feat_dim]` |
/// | `eps_f32`  | `f32`          | ε value |
/// | `n`        | `u32`          | Number of nodes |
/// | `feat_dim` | `u32`          | Feature dimension |
///
/// Launch with `grid = ceil(n * feat_dim / 256)`, `block = 256`.
pub fn gin_combine_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry gin_combine(
    .param .u64 p_self,
    .param .u64 p_aggr,
    .param .u64 p_out,
    .param .f32 eps_f32,
    .param .u32 n,
    .param .u32 feat_dim
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_self];
    ld.param.u64  %rd1, [p_aggr];
    ld.param.u64  %rd2, [p_out];
    ld.param.f32  %f0,  [eps_f32];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [feat_dim];

    // tid
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mul.lo.u32    %r6, %r0, %r1;         // n * feat_dim
    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $DONE;

    mul.wide.u32  %rd3, %r5, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];           // self_feat[tid]

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f2, [%rd5];           // aggr_feat[tid]

    // out = (1 + eps) * self + aggr
    mov.f32       %f3, 0F3F800000;        // 1.0
    add.f32       %f4, %f3, %f0;          // 1 + eps
    fma.rn.f32    %f5, %f4, %f1, %f2;   // (1+eps)*self + aggr

    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f5;

$DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 7: topk_score ────────────────────────────────────────────────────

/// TopK projection scoring with tanh.
///
/// `score[i] = tanh(dot(feat[i], p) / ||p||)`
///
/// Uses `ex2.approx.f32` to compute tanh as `2/(1+exp(-2x))-1`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_feat`    | `u64` → `f32*` | Node features `[n_nodes × feat_dim]` |
/// | `p_proj`    | `u64` → `f32*` | Projection vector `[feat_dim]` |
/// | `p_score`   | `u64` → `f32*` | Output scores `[n_nodes]` |
/// | `feat_dim`  | `u32`          | Feature dimension |
/// | `n_nodes`   | `u32`          | Number of nodes |
///
/// Launch with `grid = ceil(n_nodes / 256)`, `block = 256`.
pub fn topk_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // log2(e) = 1/ln(2) ≈ 1.4426950408889634
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    format!(
        r#"{hdr}.visible .entry topk_score(
    .param .u64 p_feat,
    .param .u64 p_proj,
    .param .u64 p_score,
    .param .u32 feat_dim,
    .param .u32 n_nodes
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p<4>;

    ld.param.u64  %rd0, [p_feat];
    ld.param.u64  %rd1, [p_proj];
    ld.param.u64  %rd2, [p_score];
    ld.param.u32  %r0,  [feat_dim];
    ld.param.u32  %r1,  [n_nodes];

    // tid = node_id
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $DONE;

    // base offset = node_id * feat_dim
    mul.lo.u32    %r6, %r5, %r0;

    // Compute dot product and norm_sq simultaneously
    mov.f32       %f0, 0F00000000;        // dot = 0
    mov.f32       %f1, 0F00000000;        // norm_sq = 0
    mov.u32       %r7, 0;                 // k = 0

$LOOP:
    setp.ge.u32   %p1, %r7, %r0;
    @%p1 bra $POSTLOOP;

    add.u32       %r8, %r6, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd3, %rd0, %rd3;
    ld.global.f32 %f2, [%rd3];           // feat[node*fd + k]

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd4, %rd1, %rd4;
    ld.global.f32 %f3, [%rd4];           // proj[k]

    fma.rn.f32    %f0, %f2, %f3, %f0;   // dot += feat * proj
    fma.rn.f32    %f1, %f3, %f3, %f1;   // norm_sq += proj^2

    add.u32       %r7, %r7, 1;
    bra           $LOOP;

$POSTLOOP:
    // norm = sqrt(norm_sq); safe divide by norm
    sqrt.approx.f32 %f4, %f1;
    // avoid div-by-zero: add tiny epsilon
    mov.f32         %f5, 0F00800000;      // 1e-38 (min normal f32)
    add.f32         %f4, %f4, %f5;
    div.approx.f32  %f6, %f0, %f4;       // x = dot / norm

    // tanh(x) = 2/(1 + exp(-2x)) - 1
    // exp(-2x) using ex2: exp(-2x) = 2^(-2x * log2e)
    mov.f32         %f7, {log2e};
    mul.f32         %f8, %f6, %f7;       // x * log2e
    neg.f32         %f9, %f8;
    add.f32         %f9, %f9, %f9;        // -2 * x * log2e
    ex2.approx.f32  %f9, %f9;            // exp(-2x)

    mov.f32         %f10, 0F3F800000;     // 1.0
    add.f32         %f9, %f10, %f9;       // 1 + exp(-2x)
    mov.f32         %f11, 0F40000000;     // 2.0
    div.approx.f32  %f9, %f11, %f9;      // 2/(1+exp(-2x))
    sub.f32         %f9, %f9, %f10;       // tanh(x)

    mul.wide.u32    %rd5, %r5, 4;
    add.u64         %rd5, %rd2, %rd5;
    st.global.f32   [%rd5], %f9;

$DONE:
    ret;
}}
"#
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    #[test]
    fn ptx_header_sm80_contains_target() {
        let h = ptx_header(80);
        assert!(h.contains(".target sm_80"));
        assert!(h.contains(".version 8.0"));
        assert!(h.contains(".address_size 64"));
    }

    #[test]
    fn ptx_header_sm90_contains_target() {
        let h = ptx_header(90);
        assert!(h.contains(".target sm_90"));
        assert!(h.contains(".version 8.4"));
    }

    #[test]
    fn ptx_header_sm120_contains_target() {
        let h = ptx_header(120);
        assert!(h.contains(".target sm_120"));
        assert!(h.contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_one() {
        // 1.0f32 bits = 0x3F800000
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
    }

    #[test]
    fn f32_hex_negative() {
        // -1.0f32 bits = 0xBF800000
        assert_eq!(f32_hex(-1.0_f32), "0FBF800000");
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = csr_spmv_ptx(sm);
            assert!(ptx.contains("csr_spmv"), "spmv missing entry for sm={sm}");
            assert!(ptx.contains(&format!("sm_{sm}")));

            let ptx = scatter_add_ptx(sm);
            assert!(
                ptx.contains("scatter_add"),
                "scatter_add missing entry for sm={sm}"
            );

            let ptx = gat_attention_ptx(sm);
            assert!(
                ptx.contains("gat_attention"),
                "gat_attention missing entry for sm={sm}"
            );

            let ptx = softmax_edge_ptx(sm);
            assert!(
                ptx.contains("softmax_edge"),
                "softmax_edge missing entry for sm={sm}"
            );

            let ptx = aggregate_mean_ptx(sm);
            assert!(
                ptx.contains("aggregate_mean"),
                "aggregate_mean missing entry for sm={sm}"
            );

            let ptx = gin_combine_ptx(sm);
            assert!(
                ptx.contains("gin_combine"),
                "gin_combine missing entry for sm={sm}"
            );

            let ptx = topk_score_ptx(sm);
            assert!(
                ptx.contains("topk_score"),
                "topk_score missing entry for sm={sm}"
            );
        }
    }

    #[test]
    fn csr_spmv_ptx_has_warp_reduction() {
        let ptx = csr_spmv_ptx(80);
        // Must contain warp-level shfl_down reduction
        assert!(ptx.contains("shfl.sync.down.b32"));
    }

    #[test]
    fn scatter_add_ptx_uses_atomic() {
        let ptx = scatter_add_ptx(80);
        assert!(ptx.contains("atom.global.add.f32"));
    }

    #[test]
    fn gat_attention_ptx_has_leaky_relu() {
        let ptx = gat_attention_ptx(80);
        // LeakyReLU is applied; we check a branch and mul sequence
        assert!(ptx.contains("setp.lt.f32"));
        assert!(ptx.contains("selp.f32"));
    }

    #[test]
    fn softmax_edge_ptx_has_exp_and_div() {
        let ptx = softmax_edge_ptx(80);
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("div.approx.f32"));
    }

    #[test]
    fn softmax_edge_ptx_scales_by_log2e_for_base_e_exp() {
        // Regression guard: the SFU `ex2.approx.f32` is base-2, so a correct
        // base-e softmax must scale `(score - max)` by log2(e) before `ex2`.
        // Without it the kernel silently computes a base-2 softmax.
        let ptx = softmax_edge_ptx(86);
        let log2e = f32_hex(std::f32::consts::LOG2_E);
        assert!(
            ptx.contains(&log2e),
            "softmax_edge must embed the log2(e) constant {log2e}"
        );
        // Two ex2 sites (sum loop + normalise loop), the scale loaded once into
        // %f5 and multiplied in at both sites (mov + 2 muls ⇒ ≥ 3 uses of %f5).
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 2);
        assert!(
            ptx.matches("%f5").count() >= 3,
            "log2(e) scale must feed both ex2 sites"
        );
    }

    #[test]
    fn gin_combine_ptx_uses_fma() {
        let ptx = gin_combine_ptx(80);
        assert!(ptx.contains("fma.rn.f32"));
        // The 1.0f32 constant should appear
        assert!(ptx.contains("3F800000"));
    }

    #[test]
    fn topk_score_ptx_uses_tanh_approx() {
        let ptx = topk_score_ptx(80);
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("sqrt.approx.f32"));
    }
}
