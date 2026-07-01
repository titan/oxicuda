fn ptx_header(sm: u32) -> String {
    let ver = match sm {
        v if v >= 100 => "8.7",
        v if v >= 90 => "8.4",
        v if v >= 80 => "8.0",
        _ => "7.5",
    };
    format!(".version {ver}\n.target sm_{sm}\n.address_size 64\n")
}

/// Pairwise L2² distances: B queries × N database vectors of dim D.
/// Thread (b, n) computes Σ_d (q\[b,d\] - x\[n,d\])^2.
pub fn l2_distance_batch_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: l2_distance_batch
// params: q[B,D], x[N,D], out[B,N], B, N, D
.visible .entry l2_distance_batch(
    .param .u64 param_q,
    .param .u64 param_x,
    .param .u64 param_out,
    .param .u32 param_B,
    .param .u32 param_N,
    .param .u32 param_D
)
{{
    .reg .u64 %q_ptr, %x_ptr, %out_ptr;
    .reg .u32 %b, %n, %d, %B, %N, %D;
    .reg .u32 %tid_x, %bid_x, %ntid_x, %bid_y, %ntid_y, %tid_y;
    .reg .f32 %acc, %qval, %xval, %diff;
    .reg .u64 %q_off, %x_off, %out_off;
    .reg .pred %p_loop_d, %p_b, %p_n;
    .reg .u32 %q_idx, %x_idx, %out_idx;

    ld.param.u64 %q_ptr, [param_q];
    ld.param.u64 %x_ptr, [param_x];
    ld.param.u64 %out_ptr, [param_out];
    ld.param.u32 %B, [param_B];
    ld.param.u32 %N, [param_N];
    ld.param.u32 %D, [param_D];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %tid_y, %tid.y;
    mov.u32 %bid_y, %ctaid.y;
    mov.u32 %ntid_y, %ntid.y;

    // b = blockIdx.y * blockDim.y + threadIdx.y
    // n = blockIdx.x * blockDim.x + threadIdx.x
    mad.lo.u32 %b, %bid_y, %ntid_y, %tid_y;
    mad.lo.u32 %n, %bid_x, %ntid_x, %tid_x;

    setp.ge.u32 %p_b, %b, %B;
    @%p_b bra DONE;
    setp.ge.u32 %p_n, %n, %N;
    @%p_n bra DONE;

    mov.f32 %acc, {zero_hex};
    mov.u32 %d, 0;

LOOP_D:
    setp.ge.u32 %p_loop_d, %d, %D;
    @%p_loop_d bra END_LOOP_D;

    // q[b,d] = q_ptr + (b*D + d) * 4
    mad.lo.u32 %q_idx, %b, %D, %d;
    cvt.u64.u32 %q_off, %q_idx;
    shl.b64 %q_off, %q_off, 2;
    add.u64 %q_off, %q_off, %q_ptr;
    ld.global.f32 %qval, [%q_off];

    // x[n,d] = x_ptr + (n*D + d) * 4
    mad.lo.u32 %x_idx, %n, %D, %d;
    cvt.u64.u32 %x_off, %x_idx;
    shl.b64 %x_off, %x_off, 2;
    add.u64 %x_off, %x_off, %x_ptr;
    ld.global.f32 %xval, [%x_off];

    sub.f32 %diff, %qval, %xval;
    fma.rn.f32 %acc, %diff, %diff, %acc;

    add.u32 %d, %d, 1;
    bra LOOP_D;

END_LOOP_D:
    // out[b,n] = out_ptr + (b*N + n) * 4
    mad.lo.u32 %out_idx, %b, %N, %n;
    cvt.u64.u32 %out_off, %out_idx;
    shl.b64 %out_off, %out_off, 2;
    add.u64 %out_off, %out_off, %out_ptr;
    st.global.f32 [%out_off], %acc;

DONE:
    ret;
}}
"#
    )
}

/// Inner product distances: B queries × N database vectors of dim D.
pub fn ip_distance_batch_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: ip_distance_batch
// params: q[B,D], x[N,D], out[B,N], B, N, D
.visible .entry ip_distance_batch(
    .param .u64 param_q,
    .param .u64 param_x,
    .param .u64 param_out,
    .param .u32 param_B,
    .param .u32 param_N,
    .param .u32 param_D
)
{{
    .reg .u64 %q_ptr, %x_ptr, %out_ptr;
    .reg .u32 %b, %n, %d, %B, %N, %D;
    .reg .u32 %tid_x, %bid_x, %ntid_x, %bid_y, %ntid_y, %tid_y;
    .reg .f32 %acc, %qval, %xval;
    .reg .u64 %q_off, %x_off, %out_off;
    .reg .pred %p_loop_d, %p_b, %p_n;
    .reg .u32 %q_idx, %x_idx, %out_idx;

    ld.param.u64 %q_ptr, [param_q];
    ld.param.u64 %x_ptr, [param_x];
    ld.param.u64 %out_ptr, [param_out];
    ld.param.u32 %B, [param_B];
    ld.param.u32 %N, [param_N];
    ld.param.u32 %D, [param_D];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %tid_y, %tid.y;
    mov.u32 %bid_y, %ctaid.y;
    mov.u32 %ntid_y, %ntid.y;

    mad.lo.u32 %b, %bid_y, %ntid_y, %tid_y;
    mad.lo.u32 %n, %bid_x, %ntid_x, %tid_x;

    setp.ge.u32 %p_b, %b, %B;
    @%p_b bra DONE;
    setp.ge.u32 %p_n, %n, %N;
    @%p_n bra DONE;

    mov.f32 %acc, {zero_hex};
    mov.u32 %d, 0;

LOOP_D:
    setp.ge.u32 %p_loop_d, %d, %D;
    @%p_loop_d bra END_LOOP_D;

    mad.lo.u32 %q_idx, %b, %D, %d;
    cvt.u64.u32 %q_off, %q_idx;
    shl.b64 %q_off, %q_off, 2;
    add.u64 %q_off, %q_off, %q_ptr;
    ld.global.f32 %qval, [%q_off];

    mad.lo.u32 %x_idx, %n, %D, %d;
    cvt.u64.u32 %x_off, %x_idx;
    shl.b64 %x_off, %x_off, 2;
    add.u64 %x_off, %x_off, %x_ptr;
    ld.global.f32 %xval, [%x_off];

    fma.rn.f32 %acc, %qval, %xval, %acc;

    add.u32 %d, %d, 1;
    bra LOOP_D;

END_LOOP_D:
    mad.lo.u32 %out_idx, %b, %N, %n;
    cvt.u64.u32 %out_off, %out_idx;
    shl.b64 %out_off, %out_off, 2;
    add.u64 %out_off, %out_off, %out_ptr;
    st.global.f32 [%out_off], %acc;

DONE:
    ret;
}}
"#
    )
}

/// Build PQ asymmetric distance table [m × ksub] for one query.
/// Block per subspace m; thread per codebook entry ksub=256.
pub fn pq_adc_table_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: pq_adc_table
// params: query[D], centroids[M,ksub,dsub], table[M,ksub], M, ksub, dsub
.visible .entry pq_adc_table(
    .param .u64 param_query,
    .param .u64 param_centroids,
    .param .u64 param_table,
    .param .u32 param_M,
    .param .u32 param_ksub,
    .param .u32 param_dsub
)
{{
    .reg .u64 %qptr, %cptr, %tptr;
    .reg .u32 %m, %k, %d, %M, %ksub, %dsub;
    .reg .u32 %tid_x, %bid_x;
    .reg .f32 %acc, %qval, %cval, %diff;
    .reg .u64 %q_off, %c_off, %t_off;
    .reg .pred %p_loop, %p_bound;
    .reg .u32 %q_idx, %c_idx, %t_idx;

    ld.param.u64 %qptr, [param_query];
    ld.param.u64 %cptr, [param_centroids];
    ld.param.u64 %tptr, [param_table];
    ld.param.u32 %M, [param_M];
    ld.param.u32 %ksub, [param_ksub];
    ld.param.u32 %dsub, [param_dsub];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %bid_x, %ctaid.x;

    // m = blockIdx.x (one block per subspace)
    // k = threadIdx.x (one thread per codebook entry)
    mov.u32 %m, %bid_x;
    mov.u32 %k, %tid_x;

    setp.ge.u32 %p_bound, %m, %M;
    @%p_bound bra DONE;
    setp.ge.u32 %p_bound, %k, %ksub;
    @%p_bound bra DONE;

    mov.f32 %acc, {zero_hex};
    mov.u32 %d, 0;

LOOP_DSUB:
    setp.ge.u32 %p_loop, %d, %dsub;
    @%p_loop bra END_LOOP;

    // query sub-vector: query[m*dsub + d]
    mad.lo.u32 %q_idx, %m, %dsub, %d;
    cvt.u64.u32 %q_off, %q_idx;
    shl.b64 %q_off, %q_off, 2;
    add.u64 %q_off, %q_off, %qptr;
    ld.global.f32 %qval, [%q_off];

    // centroid: centroids[(m*ksub + k)*dsub + d]
    mad.lo.u32 %c_idx, %m, %ksub, %k;
    mad.lo.u32 %c_idx, %c_idx, %dsub, %d;
    cvt.u64.u32 %c_off, %c_idx;
    shl.b64 %c_off, %c_off, 2;
    add.u64 %c_off, %c_off, %cptr;
    ld.global.f32 %cval, [%c_off];

    sub.f32 %diff, %qval, %cval;
    fma.rn.f32 %acc, %diff, %diff, %acc;

    add.u32 %d, %d, 1;
    bra LOOP_DSUB;

END_LOOP:
    // table[m*ksub + k]
    mad.lo.u32 %t_idx, %m, %ksub, %k;
    cvt.u64.u32 %t_off, %t_idx;
    shl.b64 %t_off, %t_off, 2;
    add.u64 %t_off, %t_off, %tptr;
    st.global.f32 [%t_off], %acc;

DONE:
    ret;
}}
"#
    )
}

/// Given a query and K candidate neighbor indices, compute L2² to each candidate.
pub fn hnsw_neighbor_eval_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: hnsw_neighbor_eval
// params: query[D], dataset[N,D], candidates[K], out_dists[K], D, K
.visible .entry hnsw_neighbor_eval(
    .param .u64 param_query,
    .param .u64 param_dataset,
    .param .u64 param_candidates,
    .param .u64 param_out,
    .param .u32 param_D,
    .param .u32 param_K
)
{{
    .reg .u64 %qptr, %dptr, %cptr, %optr;
    .reg .u32 %tidx, %D, %K, %cand_id, %d;
    .reg .f32 %acc, %qval, %dval, %diff;
    .reg .u64 %q_off, %d_off, %c_off, %o_off;
    .reg .pred %p_k, %p_d;
    .reg .u32 %d_idx;

    ld.param.u64 %qptr, [param_query];
    ld.param.u64 %dptr, [param_dataset];
    ld.param.u64 %cptr, [param_candidates];
    ld.param.u64 %optr, [param_out];
    ld.param.u32 %D, [param_D];
    ld.param.u32 %K, [param_K];

    mov.u32 %tidx, %tid.x;
    setp.ge.u32 %p_k, %tidx, %K;
    @%p_k bra DONE;

    // Load candidate index
    cvt.u64.u32 %c_off, %tidx;
    shl.b64 %c_off, %c_off, 2;
    add.u64 %c_off, %c_off, %cptr;
    ld.global.u32 %cand_id, [%c_off];

    mov.f32 %acc, {zero_hex};
    mov.u32 %d, 0;

LOOP_D:
    setp.ge.u32 %p_d, %d, %D;
    @%p_d bra END_LOOP;

    // query[d]
    cvt.u64.u32 %q_off, %d;
    shl.b64 %q_off, %q_off, 2;
    add.u64 %q_off, %q_off, %qptr;
    ld.global.f32 %qval, [%q_off];

    // dataset[cand_id, d]
    mad.lo.u32 %d_idx, %cand_id, %D, %d;
    cvt.u64.u32 %d_off, %d_idx;
    shl.b64 %d_off, %d_off, 2;
    add.u64 %d_off, %d_off, %dptr;
    ld.global.f32 %dval, [%d_off];

    sub.f32 %diff, %qval, %dval;
    fma.rn.f32 %acc, %diff, %diff, %acc;

    add.u32 %d, %d, 1;
    bra LOOP_D;

END_LOOP:
    cvt.u64.u32 %o_off, %tidx;
    shl.b64 %o_off, %o_off, 2;
    add.u64 %o_off, %o_off, %optr;
    st.global.f32 [%o_off], %acc;

DONE:
    ret;
}}
"#
    )
}

/// Assign each of B vectors to its nearest of Nc centroids.
pub fn ivf_assign_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let inf_hex = format!("0F{:08X}", f32::INFINITY.to_bits());
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: ivf_assign
// params: vectors[B,D], centroids[Nc,D], assignments[B], B, Nc, D
.visible .entry ivf_assign(
    .param .u64 param_vectors,
    .param .u64 param_centroids,
    .param .u64 param_assignments,
    .param .u32 param_B,
    .param .u32 param_Nc,
    .param .u32 param_D
)
{{
    .reg .u64 %vptr, %cptr, %aptr;
    .reg .u32 %tidx, %B, %Nc, %D;
    .reg .u32 %nc, %d, %best_id;
    .reg .f32 %best_dist, %cur_dist, %vval, %cval, %diff;
    .reg .u64 %v_off, %c_off, %a_off;
    .reg .pred %p_b, %p_nc, %p_d, %p_better;
    .reg .u32 %v_idx, %c_idx;

    ld.param.u64 %vptr, [param_vectors];
    ld.param.u64 %cptr, [param_centroids];
    ld.param.u64 %aptr, [param_assignments];
    ld.param.u32 %B, [param_B];
    ld.param.u32 %Nc, [param_Nc];
    ld.param.u32 %D, [param_D];

    mov.u32 %tidx, %tid.x;
    setp.ge.u32 %p_b, %tidx, %B;
    @%p_b bra DONE;

    mov.f32 %best_dist, {inf_hex};
    mov.u32 %best_id, 0;
    mov.u32 %nc, 0;

LOOP_NC:
    setp.ge.u32 %p_nc, %nc, %Nc;
    @%p_nc bra END_NC;

    mov.f32 %cur_dist, {zero_hex};
    mov.u32 %d, 0;

LOOP_D:
    setp.ge.u32 %p_d, %d, %D;
    @%p_d bra END_D;

    mad.lo.u32 %v_idx, %tidx, %D, %d;
    cvt.u64.u32 %v_off, %v_idx;
    shl.b64 %v_off, %v_off, 2;
    add.u64 %v_off, %v_off, %vptr;
    ld.global.f32 %vval, [%v_off];

    mad.lo.u32 %c_idx, %nc, %D, %d;
    cvt.u64.u32 %c_off, %c_idx;
    shl.b64 %c_off, %c_off, 2;
    add.u64 %c_off, %c_off, %cptr;
    ld.global.f32 %cval, [%c_off];

    sub.f32 %diff, %vval, %cval;
    fma.rn.f32 %cur_dist, %diff, %diff, %cur_dist;

    add.u32 %d, %d, 1;
    bra LOOP_D;

END_D:
    setp.lt.f32 %p_better, %cur_dist, %best_dist;
    @%p_better mov.f32 %best_dist, %cur_dist;
    @%p_better mov.u32 %best_id, %nc;

    add.u32 %nc, %nc, 1;
    bra LOOP_NC;

END_NC:
    cvt.u64.u32 %a_off, %tidx;
    shl.b64 %a_off, %a_off, 2;
    add.u64 %a_off, %a_off, %aptr;
    st.global.u32 [%a_off], %best_id;

DONE:
    ret;
}}
"#
    )
}

/// Sign of random projections Wx. Thread (b,j) computes sign(W\[j\] · x\[b\]).
pub fn lsh_random_proj_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#"{hdr}
// Kernel: lsh_random_proj
// params: x[B,D], W[K,D], out_bits[B, K/32 u32], B, K, D
.visible .entry lsh_random_proj(
    .param .u64 param_x,
    .param .u64 param_W,
    .param .u64 param_out,
    .param .u32 param_B,
    .param .u32 param_K,
    .param .u32 param_D
)
{{
    .reg .u64 %xptr, %wptr, %optr;
    .reg .u32 %tid_x, %bid_x, %ntid_x, %tid_y, %bid_y, %ntid_y;
    .reg .u32 %b, %j, %d, %B, %K, %D;
    .reg .f32 %acc, %xval, %wval;
    .reg .u64 %x_off, %w_off, %o_off;
    .reg .pred %p_b, %p_j, %p_d, %p_pos;
    .reg .u32 %x_idx, %w_idx, %o_idx;
    .reg .u32 %bit, %word_idx, %bit_idx, %mask;

    ld.param.u64 %xptr, [param_x];
    ld.param.u64 %wptr, [param_W];
    ld.param.u64 %optr, [param_out];
    ld.param.u32 %B, [param_B];
    ld.param.u32 %K, [param_K];
    ld.param.u32 %D, [param_D];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %tid_y, %tid.y;
    mov.u32 %bid_y, %ctaid.y;
    mov.u32 %ntid_y, %ntid.y;

    mad.lo.u32 %b, %bid_y, %ntid_y, %tid_y;
    mad.lo.u32 %j, %bid_x, %ntid_x, %tid_x;

    setp.ge.u32 %p_b, %b, %B;
    @%p_b bra DONE;
    setp.ge.u32 %p_j, %j, %K;
    @%p_j bra DONE;

    mov.f32 %acc, {zero_hex};
    mov.u32 %d, 0;

LOOP_D:
    setp.ge.u32 %p_d, %d, %D;
    @%p_d bra END_D;

    mad.lo.u32 %x_idx, %b, %D, %d;
    cvt.u64.u32 %x_off, %x_idx;
    shl.b64 %x_off, %x_off, 2;
    add.u64 %x_off, %x_off, %xptr;
    ld.global.f32 %xval, [%x_off];

    mad.lo.u32 %w_idx, %j, %D, %d;
    cvt.u64.u32 %w_off, %w_idx;
    shl.b64 %w_off, %w_off, 2;
    add.u64 %w_off, %w_off, %wptr;
    ld.global.f32 %wval, [%w_off];

    fma.rn.f32 %acc, %wval, %xval, %acc;

    add.u32 %d, %d, 1;
    bra LOOP_D;

END_D:
    // sign bit: 1 if acc >= 0
    setp.ge.f32 %p_pos, %acc, {zero_hex};
    mov.u32 %bit, 0;
    @%p_pos mov.u32 %bit, 1;

    // pack into output word: out[b, j/32] bit j%32
    div.u32 %word_idx, %j, 32;
    rem.u32 %bit_idx, %j, 32;
    shl.b32 %mask, %bit, %bit_idx;

    // K words per row (K/32 rounded up); use K/32 + 1 for safety
    // word offset = b * ceil(K/32) + word_idx
    // simplified: stride = (K + 31) / 32
    add.u32 %o_idx, %K, 31;
    div.u32 %o_idx, %o_idx, 32;
    mad.lo.u32 %o_idx, %b, %o_idx, %word_idx;
    cvt.u64.u32 %o_off, %o_idx;
    shl.b64 %o_off, %o_off, 2;
    add.u64 %o_off, %o_off, %optr;
    atom.global.or.b32 %o_idx, [%o_off], %mask;

DONE:
    ret;
}}
"#
    )
}

/// Top-K minimum distances from N-element array (K≤64).
///
/// One block processes one query's N distances. The block must be launched with
/// `blockDim.x = ntid` where `ntid` is a power of two in `[N, 64]`; threads with
/// `tid >= N` seed `+inf` so they sort to the bottom and never appear in the
/// top-`K` (requires `K <= N`). A full ascending bitonic sort over the shared
/// arrays leaves the `K` smallest distances (with their original indices) in
/// slots `0..K`, matching the crate's `topk::select::select_topk` (K smallest,
/// sorted ascending).
pub fn topk_select_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let inf_hex = format!("0F{:08X}", f32::INFINITY.to_bits());
    format!(
        r#"{hdr}
// Kernel: topk_select
// params: dists[N], out_dists[K], out_indices[K], N, K
// One block processes one query's N distances; blockDim.x = power-of-two in [N,64].
// Shared-memory ascending bitonic sort to find the K minimum distances.
.visible .entry topk_select(
    .param .u64 param_dists,
    .param .u64 param_out_dists,
    .param .u64 param_out_indices,
    .param .u32 param_N,
    .param .u32 param_K
)
{{
    .reg .u64 %dptr, %odptr, %oiptr, %g_off, %o_off;
    .reg .u32 %tidx, %N, %K, %n, %sbd, %sbi, %amy, %apar, %tmp;
    .reg .u32 %k, %j, %ixj, %my_idx, %b_idx;
    .reg .f32 %my_dist, %b_val, %inf;
    .reg .pred %p, %p_lt, %p_lower, %p_asc, %p_keepmin, %p_xor, %p_nkeep;
    .reg .pred %p_blt, %p_bgt, %p_t1, %p_t2, %p_take, %p_store;
    .shared .align 4 .f32 sh_dists[64];
    .shared .align 4 .u32 sh_indices[64];

    ld.param.u64 %dptr, [param_dists];
    ld.param.u64 %odptr, [param_out_dists];
    ld.param.u64 %oiptr, [param_out_indices];
    ld.param.u32 %N, [param_N];
    ld.param.u32 %K, [param_K];

    mov.u32 %tidx, %tid.x;
    mov.u32 %n, %ntid.x;
    mov.u32 %sbd, sh_dists;
    mov.u32 %sbi, sh_indices;
    mov.f32 %inf, {inf_hex};

    // Each thread loads element `tid` (or +inf if tid >= N); index = tid.
    mov.f32 %my_dist, %inf;
    mov.u32 %my_idx, %tidx;
    setp.lt.u32 %p_lt, %tidx, %N;
    @%p_lt cvt.u64.u32 %g_off, %tidx;
    @%p_lt shl.b64 %g_off, %g_off, 2;
    @%p_lt add.u64 %g_off, %g_off, %dptr;
    @%p_lt ld.global.f32 %my_dist, [%g_off];

    mad.lo.u32 %amy, %tidx, 4, %sbd;
    st.shared.f32 [%amy], %my_dist;
    mad.lo.u32 %tmp, %tidx, 4, %sbi;
    st.shared.u32 [%tmp], %my_idx;
    bar.sync 0;

    // Bitonic sort (ascending). Outer: k = 2,4,..,n. Inner: j = k/2,..,1.
    mov.u32 %k, 2;
LOOP_K:
    setp.gt.u32 %p, %k, %n;
    @%p bra END_K;
    shr.u32 %j, %k, 1;
LOOP_J:
    setp.eq.u32 %p, %j, 0;
    @%p bra END_J;

    xor.b32 %ixj, %tidx, %j;

    // Read partner's (value, index) from shared; my own pair stays in registers.
    mad.lo.u32 %apar, %ixj, 4, %sbd;
    ld.shared.f32 %b_val, [%apar];
    mad.lo.u32 %tmp, %ixj, 4, %sbi;
    ld.shared.u32 %b_idx, [%tmp];
    bar.sync 0;                          // all partner reads done before any write

    // lower = ((tid & j) == 0); ascending = ((tid & k) == 0)
    and.b32 %tmp, %tidx, %j;
    setp.eq.u32 %p_lower, %tmp, 0;
    and.b32 %tmp, %tidx, %k;
    setp.eq.u32 %p_asc, %tmp, 0;
    // keep_min = NOT(lower XOR ascending)
    xor.pred %p_xor, %p_lower, %p_asc;
    not.pred %p_keepmin, %p_xor;
    not.pred %p_nkeep, %p_keepmin;
    // take_partner = (keep_min & b<a) | (keep_max & b>a)
    setp.lt.f32 %p_blt, %b_val, %my_dist;
    setp.gt.f32 %p_bgt, %b_val, %my_dist;
    and.pred %p_t1, %p_keepmin, %p_blt;
    and.pred %p_t2, %p_nkeep, %p_bgt;
    or.pred %p_take, %p_t1, %p_t2;
    @%p_take mov.f32 %my_dist, %b_val;
    @%p_take mov.u32 %my_idx, %b_idx;

    mad.lo.u32 %amy, %tidx, 4, %sbd;
    st.shared.f32 [%amy], %my_dist;
    mad.lo.u32 %tmp, %tidx, 4, %sbi;
    st.shared.u32 [%tmp], %my_idx;
    bar.sync 0;                          // all writes done before next read

    shr.u32 %j, %j, 1;
    bra LOOP_J;
END_J:
    shl.b32 %k, %k, 1;
    bra LOOP_K;
END_K:

    // Write the K smallest (slots 0..K), already ascending.
    setp.ge.u32 %p_store, %tidx, %K;
    @%p_store bra DONE;
    mad.lo.u32 %amy, %tidx, 4, %sbd;
    ld.shared.f32 %my_dist, [%amy];
    mad.lo.u32 %tmp, %tidx, 4, %sbi;
    ld.shared.u32 %my_idx, [%tmp];

    cvt.u64.u32 %o_off, %tidx;
    shl.b64 %o_off, %o_off, 2;
    add.u64 %o_off, %o_off, %odptr;
    st.global.f32 [%o_off], %my_dist;

    cvt.u64.u32 %o_off, %tidx;
    shl.b64 %o_off, %o_off, 2;
    add.u64 %o_off, %o_off, %oiptr;
    st.global.u32 [%o_off], %my_idx;

DONE:
    ret;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kernels_nonempty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!l2_distance_batch_ptx(sm).is_empty(), "sm={sm}");
            assert!(!ip_distance_batch_ptx(sm).is_empty(), "sm={sm}");
            assert!(!pq_adc_table_ptx(sm).is_empty(), "sm={sm}");
            assert!(!hnsw_neighbor_eval_ptx(sm).is_empty(), "sm={sm}");
            assert!(!ivf_assign_ptx(sm).is_empty(), "sm={sm}");
            assert!(!lsh_random_proj_ptx(sm).is_empty(), "sm={sm}");
            assert!(!topk_select_ptx(sm).is_empty(), "sm={sm}");
        }
    }
}
