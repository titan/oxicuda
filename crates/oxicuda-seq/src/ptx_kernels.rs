//! GPU PTX kernels for Sequence Models & Structured Prediction.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised
//! on the SM version.  PTX ISA selection by SM:
//!     SM≥100 → 8.7 (Blackwell), SM≥90 → 8.4 (Hopper),
//!     SM≥80  → 8.0 (Ampere),    else → 7.5 (Turing).
//!
//! IMPORTANT: PTX kernel bodies use **string concatenation** (NOT `format!()`)
//! for sections containing `%rd`, `%r`, `%f` register names, which Rust's
//! `format!` macro would reject as malformed positional specifiers in
//! edition 2024.

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

/// HMM forward-pass kernel (log-space).
///
/// Signature: `forward_pass_kernel(alpha_prev, alpha_next, log_a, log_b_o, n_states)`
/// One thread per destination state `j`; reads `alpha_prev[i] + log_a[i*S+j]`
/// over all `i`, applies max+log-sum-exp, then adds `log_b_o[j]`.
#[must_use]
pub fn forward_pass_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry forward_pass_kernel(\n\
        .param .u64 p_alpha_prev,\n\
        .param .u64 p_alpha_next,\n\
        .param .u64 p_log_a,\n\
        .param .u64 p_log_b_o,\n\
        .param .u32 p_n_states\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_alpha_prev];\n\
        ld.param.u64  %rd1, [p_alpha_next];\n\
        ld.param.u64  %rd2, [p_log_a];\n\
        ld.param.u64  %rd3, [p_log_b_o];\n\
        ld.param.u32  %r0,  [p_n_states];\n\
    \n\
        // j = global thread id\n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $FP_DONE;\n\
    \n\
        // First pass: find max of (alpha_prev[i] + log_a[i*S + j])\n\
        mov.f32       %f0, 0fFF800000;   // -inf\n\
        mov.u32       %r5, 0;\n\
    $FP_MAX:\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $FP_SUM_INIT;\n\
        // alpha_prev[i]\n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        // log_a[i*S + j]\n\
        mul.lo.u32    %r6, %r5, %r0;\n\
        add.u32       %r6, %r6, %r4;\n\
        mul.wide.u32  %rd6, %r6, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
        add.f32       %f3, %f1, %f2;\n\
        max.f32       %f0, %f0, %f3;\n\
        add.u32       %r5, %r5, 1;\n\
        bra $FP_MAX;\n\
    \n\
    $FP_SUM_INIT:\n\
        // Second pass: accumulate exp((alpha_prev[i] + log_a[i*S+j]) - max)\n\
        mov.f32       %f4, 0f00000000;\n\
        mov.u32       %r5, 0;\n\
    $FP_SUM:\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $FP_WRITE;\n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        mul.lo.u32    %r6, %r5, %r0;\n\
        add.u32       %r6, %r6, %r4;\n\
        mul.wide.u32  %rd6, %r6, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
        add.f32       %f3, %f1, %f2;\n\
        sub.f32       %f3, %f3, %f0;\n\
        ex2.approx.f32 %f3, %f3;\n\
        add.f32       %f4, %f4, %f3;\n\
        add.u32       %r5, %r5, 1;\n\
        bra $FP_SUM;\n\
    \n\
    $FP_WRITE:\n\
        // result = max + log(sum) + log_b_o[j]\n\
        lg2.approx.f32 %f4, %f4;\n\
        add.f32       %f4, %f4, %f0;\n\
        mul.wide.u32  %rd4, %r4, 4;\n\
        add.u64       %rd5, %rd3, %rd4;\n\
        ld.global.f32 %f5, [%rd5];\n\
        add.f32       %f4, %f4, %f5;\n\
        add.u64       %rd6, %rd1, %rd4;\n\
        st.global.f32 [%rd6], %f4;\n\
    \n\
    $FP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Viterbi step kernel (log-space, with argmax storage).
///
/// Signature: `viterbi_step_kernel(delta_prev, delta_next, log_a, log_b_o, psi, n_states)`
/// Each thread computes one destination `j`: δ_t(j) = max_i(δ_{t-1}(i) + log A_ij) + log B_j(o_t).
/// Argmax index is stored in `psi[j]` as `s32`.
#[must_use]
pub fn viterbi_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry viterbi_step_kernel(\n\
        .param .u64 p_delta_prev,\n\
        .param .u64 p_delta_next,\n\
        .param .u64 p_log_a,\n\
        .param .u64 p_log_b_o,\n\
        .param .u64 p_psi,\n\
        .param .u32 p_n_states\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .s32  %sr<4>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_delta_prev];\n\
        ld.param.u64  %rd1, [p_delta_next];\n\
        ld.param.u64  %rd2, [p_log_a];\n\
        ld.param.u64  %rd3, [p_log_b_o];\n\
        ld.param.u64  %rd4, [p_psi];\n\
        ld.param.u32  %r0,  [p_n_states];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $VS_DONE;\n\
    \n\
        mov.f32       %f0, 0fFF800000;   // best = -inf\n\
        mov.s32       %sr0, -1;          // argmax\n\
        mov.u32       %r5, 0;\n\
    $VS_LOOP:\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $VS_WRITE;\n\
        // delta_prev[i]\n\
        mul.wide.u32  %rd5, %r5, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
        // log_a[i*S + j]\n\
        mul.lo.u32    %r6, %r5, %r0;\n\
        add.u32       %r6, %r6, %r4;\n\
        mul.wide.u32  %rd7, %r6, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        ld.global.f32 %f2, [%rd8];\n\
        add.f32       %f3, %f1, %f2;\n\
        setp.gt.f32   %p1, %f3, %f0;\n\
        @%p1 mov.f32  %f0, %f3;\n\
        @%p1 cvt.s32.u32 %sr0, %r5;\n\
        add.u32       %r5, %r5, 1;\n\
        bra $VS_LOOP;\n\
    \n\
    $VS_WRITE:\n\
        // delta_next[j] = best + log_b_o[j]\n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd3, %rd5;\n\
        ld.global.f32 %f4, [%rd6];\n\
        add.f32       %f0, %f0, %f4;\n\
        add.u64       %rd7, %rd1, %rd5;\n\
        st.global.f32 [%rd7], %f0;\n\
        // psi[j] = argmax\n\
        add.u64       %rd8, %rd4, %rd5;\n\
        st.global.s32 [%rd8], %sr0;\n\
    \n\
    $VS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// CRF feature-score kernel.
///
/// Signature: `crf_features_kernel(emit, trans, x_feat, score, t, n_labels, n_features)`
/// Computes per-(label,prev_label) score at time `t`: `emit[y]·x_feat[t] + trans[prev,y]`.
#[must_use]
pub fn crf_features_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry crf_features_kernel(\n\
        .param .u64 p_emit,\n\
        .param .u64 p_trans,\n\
        .param .u64 p_x_feat,\n\
        .param .u64 p_score,\n\
        .param .u32 p_t,\n\
        .param .u32 p_n_labels,\n\
        .param .u32 p_n_features\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<14>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_emit];\n\
        ld.param.u64  %rd1, [p_trans];\n\
        ld.param.u64  %rd2, [p_x_feat];\n\
        ld.param.u64  %rd3, [p_score];\n\
        ld.param.u32  %r0,  [p_t];\n\
        ld.param.u32  %r1,  [p_n_labels];\n\
        ld.param.u32  %r2,  [p_n_features];\n\
    \n\
        // y_prev = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r3, %ntid.y;\n\
        mov.u32       %r4, %ctaid.y;\n\
        mov.u32       %r5, %tid.y;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
        // y_cur = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r7, %ntid.x;\n\
        mov.u32       %r8, %ctaid.x;\n\
        mov.u32       %r9, %tid.x;\n\
        mad.lo.u32    %r10, %r7, %r8, %r9;\n\
    \n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $CF_DONE;\n\
        setp.ge.u32   %p0, %r10, %r1;\n\
        @%p0 bra $CF_DONE;\n\
    \n\
        // Emission score: dot(emit[y_cur,:], x_feat[t,:])\n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r11, 0;\n\
    $CF_EMIT:\n\
        setp.ge.u32   %p0, %r11, %r2;\n\
        @%p0 bra $CF_TRANS;\n\
        // emit[y_cur * n_features + k]\n\
        mul.lo.u32    %r12, %r10, %r2;\n\
        add.u32       %r12, %r12, %r11;\n\
        mul.wide.u32  %rd4, %r12, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        // x_feat[t * n_features + k]\n\
        mul.lo.u32    %r13, %r0, %r2;\n\
        add.u32       %r13, %r13, %r11;\n\
        mul.wide.u32  %rd6, %r13, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
        add.u32       %r11, %r11, 1;\n\
        bra $CF_EMIT;\n\
    \n\
    $CF_TRANS:\n\
        // Transition score: trans[y_prev * n_labels + y_cur]\n\
        mul.lo.u32    %r12, %r6, %r1;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd4, %r12, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.f32 %f3, [%rd5];\n\
        add.f32       %f0, %f0, %f3;\n\
    \n\
        // score[y_prev * n_labels + y_cur] = f0\n\
        add.u64       %rd6, %rd3, %rd4;\n\
        st.global.f32 [%rd6], %f0;\n\
    \n\
    $CF_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Beam top-k partial-sort kernel (one-pass rank approximation).
///
/// Signature: `beam_topk_kernel(scores, rank, n, k)`
/// Each thread computes how many other scores are strictly greater (its rank);
/// threads whose rank < k are marked surviving (`rank[tid] = rank`); others get `-1`.
#[must_use]
pub fn beam_topk_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry beam_topk_kernel(\n\
        .param .u64 p_scores,\n\
        .param .u64 p_rank,\n\
        .param .u32 p_n,\n\
        .param .u32 p_k\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .s32  %sr<4>;\n\
        .reg .f32  %f<4>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_scores];\n\
        ld.param.u64  %rd1, [p_rank];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_k];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $BK_DONE;\n\
    \n\
        // my score\n\
        mul.wide.u32  %rd2, %r5, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        mov.u32       %r6, 0;     // rank counter\n\
        mov.u32       %r7, 0;     // loop index\n\
    $BK_LOOP:\n\
        setp.ge.u32   %p0, %r7, %r0;\n\
        @%p0 bra $BK_WRITE;\n\
        mul.wide.u32  %rd4, %r7, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        setp.gt.f32   %p1, %f1, %f0;\n\
        @%p1 add.u32  %r6, %r6, 1;\n\
        add.u32       %r7, %r7, 1;\n\
        bra $BK_LOOP;\n\
    \n\
    $BK_WRITE:\n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        mov.s32       %sr0, -1;\n\
        @!%p0 cvt.s32.u32 %sr0, %r6;\n\
        add.u64       %rd6, %rd1, %rd2;\n\
        st.global.s32 [%rd6], %sr0;\n\
    \n\
    $BK_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Edit-distance anti-diagonal cell update kernel.
///
/// Signature: `edit_dist_kernel(dp, a_chars, b_chars, n_a, n_b, diag)`
/// On anti-diagonal `diag`, each thread updates one cell `dp[i*(n_b+1)+j]`
/// using the standard Levenshtein recurrence.
#[must_use]
pub fn edit_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry edit_dist_kernel(\n\
        .param .u64 p_dp,\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u32 p_n_a,\n\
        .param .u32 p_n_b,\n\
        .param .u32 p_diag\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<14>;\n\
        .reg .s32  %sr<6>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_dp];\n\
        ld.param.u64  %rd1, [p_a];\n\
        ld.param.u64  %rd2, [p_b];\n\
        ld.param.u32  %r0,  [p_n_a];\n\
        ld.param.u32  %r1,  [p_n_b];\n\
        ld.param.u32  %r2,  [p_diag];\n\
    \n\
        // i = tid + 1; j = diag - i\n\
        mov.u32       %r3, %ntid.x;\n\
        mov.u32       %r4, %ctaid.x;\n\
        mov.u32       %r5, %tid.x;\n\
        mad.lo.u32    %r6, %r3, %r4, %r5;\n\
        add.u32       %r7, %r6, 1;            // i\n\
        sub.u32       %r8, %r2, %r7;          // j\n\
        setp.gt.u32   %p0, %r7, %r0;\n\
        @%p0 bra $ED_DONE;\n\
        setp.eq.u32   %p0, %r8, 0;\n\
        @%p0 bra $ED_DONE;\n\
        setp.gt.u32   %p0, %r8, %r1;\n\
        @%p0 bra $ED_DONE;\n\
    \n\
        // load a[i-1], b[j-1]\n\
        sub.u32       %r9, %r7, 1;\n\
        mul.wide.u32  %rd3, %r9, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.s32 %sr0, [%rd4];\n\
        sub.u32       %r10, %r8, 1;\n\
        mul.wide.u32  %rd5, %r10, 4;\n\
        add.u64       %rd6, %rd2, %rd5;\n\
        ld.global.s32 %sr1, [%rd6];\n\
    \n\
        // cost: 0 if eq, else 1\n\
        setp.eq.s32   %p1, %sr0, %sr1;\n\
        mov.s32       %sr2, 1;\n\
        @%p1 mov.s32  %sr2, 0;\n\
    \n\
        // Read 3 neighbours from dp\n\
        // dp[(i-1)*(n_b+1) + j]\n\
        add.u32       %r11, %r1, 1;\n\
        mul.lo.u32    %r12, %r9, %r11;\n\
        add.u32       %r12, %r12, %r8;\n\
        mul.wide.u32  %rd7, %r12, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.s32 %sr3, [%rd8];\n\
        // dp[i*(n_b+1) + (j-1)]\n\
        mul.lo.u32    %r12, %r7, %r11;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd7, %r12, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.s32 %sr4, [%rd8];\n\
        // dp[(i-1)*(n_b+1) + (j-1)]\n\
        mul.lo.u32    %r12, %r9, %r11;\n\
        add.u32       %r12, %r12, %r10;\n\
        mul.wide.u32  %rd7, %r12, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.s32 %sr5, [%rd8];\n\
    \n\
        // best = min(sr3+1, sr4+1, sr5+sr2)\n\
        add.s32       %sr3, %sr3, 1;\n\
        add.s32       %sr4, %sr4, 1;\n\
        add.s32       %sr5, %sr5, %sr2;\n\
        min.s32       %sr3, %sr3, %sr4;\n\
        min.s32       %sr3, %sr3, %sr5;\n\
    \n\
        // write dp[i*(n_b+1)+j]\n\
        mul.lo.u32    %r12, %r7, %r11;\n\
        add.u32       %r12, %r12, %r8;\n\
        mul.wide.u32  %rd7, %r12, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        st.global.s32 [%rd8], %sr3;\n\
    \n\
    $ED_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Kalman predict step kernel (matrix-vector + covariance update).
///
/// Signature: `kalman_predict_kernel(x, x_pred, A, P, P_pred, Q, n)`
/// Computes x_pred = A·x and P_pred = A·P·Aᵀ + Q.  Each thread handles one row.
#[must_use]
pub fn kalman_predict_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry kalman_predict_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_x_pred,\n\
        .param .u64 p_a,\n\
        .param .u64 p_p,\n\
        .param .u64 p_p_pred,\n\
        .param .u64 p_q,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<14>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_x_pred];\n\
        ld.param.u64  %rd2, [p_a];\n\
        ld.param.u64  %rd3, [p_p];\n\
        ld.param.u64  %rd4, [p_p_pred];\n\
        ld.param.u64  %rd5, [p_q];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $KP_DONE;\n\
    \n\
        // x_pred[i] = sum_k A[i,k] * x[k]\n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r5, 0;\n\
    $KP_VEC:\n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $KP_VEC_WR;\n\
        // A[i*n + k]\n\
        mul.lo.u32    %r6, %r4, %r0;\n\
        add.u32       %r6, %r6, %r5;\n\
        mul.wide.u32  %rd6, %r6, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        ld.global.f32 %f1, [%rd7];\n\
        // x[k]\n\
        mul.wide.u32  %rd8, %r5, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f2, [%rd9];\n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
        add.u32       %r5, %r5, 1;\n\
        bra $KP_VEC;\n\
    \n\
    $KP_VEC_WR:\n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.f32 [%rd7], %f0;\n\
    \n\
        // P_pred[i,j] = sum_{k,l} A[i,k] P[k,l] A[j,l] + Q[i,j]\n\
        // One thread handles row i, all j.\n\
        mov.u32       %r7, 0;\n\
    $KP_J:\n\
        setp.ge.u32   %p0, %r7, %r0;\n\
        @%p0 bra $KP_DONE;\n\
        mov.f32       %f3, 0f00000000;\n\
        mov.u32       %r8, 0;\n\
    $KP_K:\n\
        setp.ge.u32   %p0, %r8, %r0;\n\
        @%p0 bra $KP_K_DONE;\n\
        mov.f32       %f4, 0f00000000;\n\
        mov.u32       %r9, 0;\n\
    $KP_L:\n\
        setp.ge.u32   %p0, %r9, %r0;\n\
        @%p0 bra $KP_L_DONE;\n\
        // P[k*n + l]\n\
        mul.lo.u32    %r10, %r8, %r0;\n\
        add.u32       %r10, %r10, %r9;\n\
        mul.wide.u32  %rd6, %r10, 4;\n\
        add.u64       %rd7, %rd3, %rd6;\n\
        ld.global.f32 %f5, [%rd7];\n\
        // A[j*n + l]\n\
        mul.lo.u32    %r11, %r7, %r0;\n\
        add.u32       %r11, %r11, %r9;\n\
        mul.wide.u32  %rd8, %r11, 4;\n\
        add.u64       %rd9, %rd2, %rd8;\n\
        ld.global.f32 %f6, [%rd9];\n\
        fma.rn.f32    %f4, %f5, %f6, %f4;\n\
        add.u32       %r9, %r9, 1;\n\
        bra $KP_L;\n\
    \n\
    $KP_L_DONE:\n\
        // A[i*n + k]\n\
        mul.lo.u32    %r10, %r4, %r0;\n\
        add.u32       %r10, %r10, %r8;\n\
        mul.wide.u32  %rd6, %r10, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        ld.global.f32 %f7, [%rd7];\n\
        fma.rn.f32    %f3, %f7, %f4, %f3;\n\
        add.u32       %r8, %r8, 1;\n\
        bra $KP_K;\n\
    \n\
    $KP_K_DONE:\n\
        // P_pred[i*n+j] = f3 + Q[i*n+j]\n\
        mul.lo.u32    %r10, %r4, %r0;\n\
        add.u32       %r10, %r10, %r7;\n\
        mul.wide.u32  %rd6, %r10, 4;\n\
        add.u64       %rd7, %rd5, %rd6;\n\
        ld.global.f32 %f8, [%rd7];\n\
        add.f32       %f3, %f3, %f8;\n\
        add.u64       %rd9, %rd4, %rd6;\n\
        st.global.f32 [%rd9], %f3;\n\
        add.u32       %r7, %r7, 1;\n\
        bra $KP_J;\n\
    \n\
    $KP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Gibbs-sampling Ising MRF kernel: per-site conditional resample given neighbours.
///
/// Signature: `mrf_gibbs_kernel(spins, h, j, n_rows, n_cols, seed)`
/// Each thread samples its spin given a uniform draw from the inline LCG seeded
/// by `seed ^ tid`.
#[must_use]
pub fn mrf_gibbs_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry mrf_gibbs_kernel(\n\
        .param .u64 p_spins,\n\
        .param .f32 p_h,\n\
        .param .f32 p_j,\n\
        .param .u32 p_n_rows,\n\
        .param .u32 p_n_cols,\n\
        .param .u64 p_seed\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .s32  %sr<8>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0, %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_spins];\n\
        ld.param.f32  %f0,  [p_h];\n\
        ld.param.f32  %f1,  [p_j];\n\
        ld.param.u32  %r0,  [p_n_rows];\n\
        ld.param.u32  %r1,  [p_n_cols];\n\
        ld.param.u64  %rd1, [p_seed];\n\
    \n\
        // i = blockIdx.y * blockDim.y + threadIdx.y\n\
        mov.u32       %r2, %ntid.y;\n\
        mov.u32       %r3, %ctaid.y;\n\
        mov.u32       %r4, %tid.y;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
        // j = blockIdx.x * blockDim.x + threadIdx.x\n\
        mov.u32       %r6, %ntid.x;\n\
        mov.u32       %r7, %ctaid.x;\n\
        mov.u32       %r8, %tid.x;\n\
        mad.lo.u32    %r9, %r6, %r7, %r8;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $MG_DONE;\n\
        setp.ge.u32   %p0, %r9, %r1;\n\
        @%p0 bra $MG_DONE;\n\
    \n\
        // Sum of neighbour spins (with bounds checks)\n\
        mov.s32       %sr0, 0;\n\
        // up\n\
        setp.eq.u32   %p1, %r5, 0;\n\
        @%p1 bra $MG_NB1;\n\
        sub.u32       %r10, %r5, 1;\n\
        mul.lo.u32    %r11, %r10, %r1;\n\
        add.u32       %r11, %r11, %r9;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.s32 %sr1, [%rd3];\n\
        add.s32       %sr0, %sr0, %sr1;\n\
    $MG_NB1:\n\
        // down\n\
        add.u32       %r10, %r5, 1;\n\
        setp.ge.u32   %p1, %r10, %r0;\n\
        @%p1 bra $MG_NB2;\n\
        mul.lo.u32    %r11, %r10, %r1;\n\
        add.u32       %r11, %r11, %r9;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.s32 %sr1, [%rd3];\n\
        add.s32       %sr0, %sr0, %sr1;\n\
    $MG_NB2:\n\
        // left\n\
        setp.eq.u32   %p1, %r9, 0;\n\
        @%p1 bra $MG_NB3;\n\
        sub.u32       %r10, %r9, 1;\n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.s32 %sr1, [%rd3];\n\
        add.s32       %sr0, %sr0, %sr1;\n\
    $MG_NB3:\n\
        // right\n\
        add.u32       %r10, %r9, 1;\n\
        setp.ge.u32   %p1, %r10, %r1;\n\
        @%p1 bra $MG_FIELD;\n\
        mul.lo.u32    %r11, %r5, %r1;\n\
        add.u32       %r11, %r11, %r10;\n\
        mul.wide.u32  %rd2, %r11, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.s32 %sr1, [%rd3];\n\
        add.s32       %sr0, %sr0, %sr1;\n\
    \n\
    $MG_FIELD:\n\
        // field = j * sum + h\n\
        cvt.rn.f32.s32 %f2, %sr0;\n\
        mul.f32       %f3, %f1, %f2;\n\
        add.f32       %f3, %f3, %f0;\n\
    \n\
        // p_up = 1 / (1 + exp(-2*field))\n\
        mov.f32       %f4, 0fC0000000;       // -2\n\
        mul.f32       %f5, %f4, %f3;\n\
        ex2.approx.f32 %f5, %f5;             // exp2(...)\n\
        mov.f32       %f6, 0f3F800000;       // 1.0\n\
        add.f32       %f5, %f5, %f6;\n\
        div.rn.f32    %f7, %f6, %f5;\n\
    \n\
        // Inline LCG: seed ^ (row * n_cols + col)\n\
        mul.lo.u32    %r12, %r5, %r1;\n\
        add.u32       %r12, %r12, %r9;\n\
        cvt.u64.u32   %rd4, %r12;\n\
        xor.b64       %rd5, %rd4, %rd1;\n\
        mov.u64       %rd6, 6364136223846793005;\n\
        mul.lo.u64    %rd5, %rd5, %rd6;\n\
        mov.u64       %rd6, 1442695040888963407;\n\
        add.u64       %rd5, %rd5, %rd6;\n\
        shr.u64       %rd7, %rd5, 32;\n\
        cvt.u32.u64   %r13, %rd7;\n\
        // u = (high32 >> 8) / 2^24\n\
        shr.u32       %r14, %r13, 8;\n\
        cvt.rn.f32.u32 %f8, %r14;\n\
        mov.f32       %f9, 0f33800000;       // 1 / 2^24\n\
        mul.f32       %f8, %f8, %f9;\n\
    \n\
        // s = (u < p_up) ? +1 : -1\n\
        setp.lt.f32   %p1, %f8, %f7;\n\
        mov.s32       %sr2, -1;\n\
        @%p1 mov.s32  %sr2, 1;\n\
    \n\
        // store\n\
        mul.lo.u32    %r15, %r5, %r1;\n\
        add.u32       %r15, %r15, %r9;\n\
        mul.wide.u32  %rd8, %r15, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        st.global.s32 [%rd9], %sr2;\n\
    \n\
    $MG_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(89).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
    }

    #[test]
    fn all_kernels_non_empty() {
        type KernelFn = fn(u32) -> String;
        let kernels: &[(&str, KernelFn)] = &[
            ("forward_pass", forward_pass_ptx),
            ("viterbi_step", viterbi_step_ptx),
            ("crf_features", crf_features_ptx),
            ("beam_topk", beam_topk_ptx),
            ("edit_dist", edit_dist_ptx),
            ("kalman_predict", kalman_predict_ptx),
            ("mrf_gibbs", mrf_gibbs_ptx),
        ];
        let sms = [75u32, 80, 86, 89, 90, 100];
        for &sm in &sms {
            for &(name, f) in kernels {
                let s = f(sm);
                assert!(!s.is_empty(), "{name} sm{sm} empty");
                assert!(
                    s.contains(".visible .entry"),
                    "{name} sm{sm} missing .visible .entry"
                );
            }
        }
    }
}
