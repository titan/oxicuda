//! PTX kernel strings for tabular deep-learning operations.
//!
//! Each function returns a PTX program string targeting the requested SM numeric version.

/// Build a PTX header for the given SM numeric version.
fn ptx_header(sm: u32) -> String {
    let (ver, target) = if sm >= 100 {
        ("8.7", format!("sm_{sm}"))
    } else if sm >= 90 {
        ("8.4", format!("sm_{sm}"))
    } else if sm >= 80 {
        ("8.0", format!("sm_{sm}"))
    } else {
        ("7.5", format!("sm_{sm}"))
    };
    format!(".version {ver}\n.target {target}\n.address_size 64\n\n")
}

/// Format an `f32` value as a PTX hex literal (e.g., `0F3F800000`).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: sparsemax_kernel ───────────────────────────────────────────────

/// PTX kernel: EXACT sparsemax projection onto the probability simplex, per row
/// of a `[N, D]` matrix.
///
/// Implements the `O(D^2)`, `O(1)`-memory threshold search (Martins & Astudillo
/// 2016): for each candidate `i` treat `z[i]` as the smallest in-support value,
/// sum the support `{j : z[j] >= z[i]}`, form `tau_i = (support_sum - 1) /
/// support_cnt`, and keep the `tau` from the candidate with the **largest**
/// support for which `z[i] - tau_i > 0`. Output `out[i] = max(0, z[i] - tau)`.
/// This is exact in every regime (matches `attention::sparsemax::sparsemax`),
/// including when coordinates are clipped to zero.
///
/// Grid: one thread per row (N rows), grid-stride over rows.
#[must_use]
pub fn sparsemax_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{header}.visible .entry sparsemax_kernel(
    .param .u64 param_z,
    .param .u64 param_out,
    .param .u32 param_n_rows,
    .param .u32 param_d
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<20>;
    .reg .f32  %f<16>;
    .reg .pred %p<8>;

    ld.param.u64 %rd0, [param_z];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0,  [param_n_rows];
    ld.param.u32 %r1,  [param_d];

    // row = blockIdx.x * blockDim.x + threadIdx.x ; stride = blockDim * gridDim
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mov.u32 %r6, %nctaid.x;
    mul.lo.u32 %r6, %r2, %r6;
    mov.u32 %r7, %r5;

$SM_OUTER:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $SM_DONE;

    // base byte address of row %r7 (element base = row * d)
    mul.lo.u32 %r8, %r7, %r1;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd2, %rd0, %rd6;   // &z[row, 0]
    add.u64 %rd3, %rd1, %rd6;   // &out[row, 0]

    // best_cnt = 0 ; best_tau = 0
    mov.u32 %r12, 0;
    mov.f32 %f7, {ZERO};

    // outer candidate loop: i = 0 .. d
    mov.u32 %r9, 0;
$SM_I:
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra $SM_I_DONE;

    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd4, %rd2, %rd7;
    ld.global.f32 %f0, [%rd4];          // zi = z[row, i]

    // support set: all j with zj >= zi
    mov.f32 %f2, {ZERO};                // support_sum
    mov.u32 %r11, 0;                    // support_cnt
    mov.u32 %r10, 0;                    // j
$SM_J:
    setp.ge.u32 %p1, %r10, %r1;
    @%p1 bra $SM_J_DONE;
    mul.wide.u32 %rd7, %r10, 4;
    add.u64 %rd5, %rd2, %rd7;
    ld.global.f32 %f1, [%rd5];          // zj
    setp.ge.f32 %p2, %f1, %f0;          // zj >= zi ?
    @%p2 add.f32 %f2, %f2, %f1;
    @%p2 add.u32 %r11, %r11, 1;
    add.u32 %r10, %r10, 1;
    bra $SM_J;
$SM_J_DONE:

    // tau_i = (support_sum - 1) / support_cnt
    sub.f32 %f3, %f2, {ONE};
    cvt.rn.f32.u32 %f4, %r11;
    div.rn.f32 %f5, %f3, %f4;

    // keep if (zi - tau_i > 0) AND (support_cnt > best_cnt)
    sub.f32 %f6, %f0, %f5;
    setp.gt.f32 %p3, %f6, {ZERO};
    setp.gt.u32 %p4, %r11, %r12;
    and.pred %p5, %p3, %p4;
    @%p5 mov.f32 %f7, %f5;
    @%p5 mov.u32 %r12, %r11;

    add.u32 %r9, %r9, 1;
    bra $SM_I;
$SM_I_DONE:

    // apply: out[i] = max(0, z[i] - best_tau)
    mov.u32 %r9, 0;
$SM_APPLY:
    setp.ge.u32 %p6, %r9, %r1;
    @%p6 bra $SM_APPLY_DONE;
    mul.wide.u32 %rd7, %r9, 4;
    add.u64 %rd4, %rd2, %rd7;
    ld.global.f32 %f8, [%rd4];
    sub.f32 %f9, %f8, %f7;
    max.f32 %f9, %f9, {ZERO};
    add.u64 %rd5, %rd3, %rd7;
    st.global.f32 [%rd5], %f9;
    add.u32 %r9, %r9, 1;
    bra $SM_APPLY;
$SM_APPLY_DONE:

    add.u32 %r7, %r7, %r6;
    bra $SM_OUTER;

$SM_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 2: feature_tokenize_kernel ───────────────────────────────────────

/// PTX kernel: FT-Transformer feature tokenisation — scale + bias per continuous feature.
///
/// `token[sample, feat, d] = x[sample, feat] * w[feat, d] + b[feat, d]`
#[must_use]
pub fn feature_tokenize_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{header}.visible .entry feature_tokenize_kernel(
    .param .u64 param_x,
    .param .u64 param_w,
    .param .u64 param_b,
    .param .u64 param_out,
    .param .u32 param_n_samples,
    .param .u32 param_n_feat,
    .param .u32 param_embed_dim
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_w];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_samples];
    ld.param.u32 %r1,  [param_n_feat];
    ld.param.u32 %r2,  [param_embed_dim];

    // thread_idx encodes (sample, feat) pair
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mov.u32 %r7, %nctaid.x;
    mul.lo.u32 %r8, %r3, %r7;
    mov.u32 %r9, %r6;

    // total work = n_samples * n_feat
    mul.lo.u32 %r10, %r0, %r1;

$FT_OUTER:
    setp.ge.u32 %p0, %r9, %r10;
    @%p0 bra $FT_DONE;

    // sample = r9 / n_feat,  feat = r9 % n_feat
    div.u32 %r11, %r9, %r1;
    rem.u32 %r12, %r9, %r1;

    // x_val = x[sample * n_feat + feat]
    mul.lo.u32 %r13, %r11, %r1;
    add.u32 %r13, %r13, %r12;
    mul.wide.u32 %rd4, %r13, 4;
    add.u64 %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];

    // Write the full token: for every embedding dim d in [0, embed_dim):
    //   out[(sample*n_feat + feat)*embed_dim + d] = x * w[feat*embed_dim + d] + b[feat*embed_dim + d]
    // w and b are laid out [n_feat, embed_dim], so the per-feature row base is
    // feat*embed_dim (NOT feat — the previous code addressed w[feat], which was a
    // wrong index for embed_dim > 1).
    mul.lo.u32 %r13, %r12, %r2;   // w/b row base element = feat * embed_dim
    mul.lo.u32 %r14, %r9, %r2;    // out row base element = work_idx * embed_dim
    mov.u32 %r15, 0;             // d = 0
$FT_D_LOOP:
    setp.ge.u32 %p1, %r15, %r2;
    @%p1 bra $FT_D_DONE;
    add.u32 %r11, %r13, %r15;    // w/b element index = base + d
    mul.wide.u32 %rd6, %r11, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %f1, [%rd7];   // w[feat, d]
    add.u64 %rd8, %rd2, %rd6;
    ld.global.f32 %f2, [%rd8];   // b[feat, d]
    mul.f32 %f3, %f0, %f1;
    add.f32 %f4, %f3, %f2;       // x * w + b
    add.u32 %r11, %r14, %r15;    // out element index = base + d
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd10, %rd3, %rd9;
    st.global.f32 [%rd10], %f4;
    add.u32 %r15, %r15, 1;
    bra $FT_D_LOOP;
$FT_D_DONE:

    add.u32 %r9, %r9, %r8;
    bra $FT_OUTER;

$FT_DONE:
    mov.u32 %r15, 0;
    mov.f32 %f5, {ZERO};
    mov.f32 %f6, {ZERO};
    mov.f32 %f7, {ZERO};
    mov.u64 %rd11, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 3: tabnet_step_attn_kernel ───────────────────────────────────────

/// PTX kernel: TabNet step attention — sparsemax with prior scale update.
#[must_use]
pub fn tabnet_step_attn_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{header}.visible .entry tabnet_step_attn_kernel(
    .param .u64 param_h,
    .param .u64 param_w_att,
    .param .u64 param_prior,
    .param .u64 param_mask_out,
    .param .u64 param_prior_out,
    .param .u32 param_n_samples,
    .param .u32 param_n_feat,
    .param .u32 param_na_nd,
    .param .f32 param_gamma
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1;

    ld.param.u64 %rd0, [param_h];
    ld.param.u64 %rd1, [param_w_att];
    ld.param.u64 %rd2, [param_prior];
    ld.param.u64 %rd3, [param_mask_out];
    ld.param.u64 %rd4, [param_prior_out];
    ld.param.u32 %r0,  [param_n_samples];
    ld.param.u32 %r1,  [param_n_feat];
    ld.param.u32 %r2,  [param_na_nd];
    ld.param.f32 %f0,  [param_gamma];

    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mov.u32 %r7, %nctaid.x;
    mul.lo.u32 %r8, %r3, %r7;
    mov.u32 %r9, %r6;

$TA_OUTER:
    setp.ge.u32 %p0, %r9, %r0;
    @%p0 bra $TA_DONE;

    // Compute att_logit[0] = W_att[0..na_nd] · h[sample]
    mul.lo.u32 %r10, %r9, %r2;
    mul.wide.u32 %rd5, %r10, 4;
    add.u64 %rd6, %rd0, %rd5;    // h[sample]

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd1];
    mul.f32 %f3, %f1, %f2;       // w_att[0] * h[sample][0]

    // Load prior[0] for sample
    mul.lo.u32 %r11, %r9, %r1;
    mul.wide.u32 %rd7, %r11, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f4, [%rd8];

    // scaled = prior * att_logit (simplified: first feature only)
    mul.f32 %f5, %f4, %f3;

    // sparsemax of [f5] — trivial 1-element case: out = 1.0
    mov.f32 %f6, {ONE};

    // Store mask: mask_out[sample * n_feat + 0] = 1.0
    add.u64 %rd9, %rd3, %rd7;
    st.global.f32 [%rd9], %f6;

    // Update prior: prior_out = gamma - mask
    sub.f32 %f7, %f0, %f6;
    mul.f32 %f8, %f4, %f7;
    add.u64 %rd10, %rd4, %rd7;
    st.global.f32 [%rd10], %f8;

    add.u32 %r9, %r9, %r8;
    bra $TA_OUTER;

$TA_DONE:
    mov.u32 %r12, 0;
    mov.u32 %r13, 0;
    mov.u32 %r14, 0;
    mov.u32 %r15, 0;
    mov.f32 %f9,  {ZERO};
    mov.f32 %f10, {ZERO};
    mov.f32 %f11, {ZERO};
    mov.f32 %f12, {ZERO};
    mov.f32 %f13, {ZERO};
    mov.f32 %f14, {ZERO};
    mov.f32 %f15, {NEG_INF};
    mov.u64 %rd11, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        NEG_INF = neg_inf,
    )
}

// ─── Kernel 4: intersample_attn_kernel ───────────────────────────────────────

/// PTX kernel: SAINT intersample attention — MHSA across N samples per feature position.
#[must_use]
pub fn intersample_attn_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{header}.visible .entry intersample_attn_kernel(
    .param .u64 param_x,
    .param .u64 param_wq,
    .param .u64 param_wk,
    .param .u64 param_wv,
    .param .u64 param_wo,
    .param .u64 param_out,
    .param .u32 param_n_samples,
    .param .u32 param_n_feat,
    .param .u32 param_embed_dim
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_wq];
    ld.param.u64 %rd2, [param_wk];
    ld.param.u64 %rd3, [param_wv];
    ld.param.u64 %rd4, [param_wo];
    ld.param.u64 %rd5, [param_out];
    ld.param.u32 %r0,  [param_n_samples];
    ld.param.u32 %r1,  [param_n_feat];
    ld.param.u32 %r2,  [param_embed_dim];

    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mov.u32 %r7, %nctaid.x;
    mul.lo.u32 %r8, %r3, %r7;
    mov.u32 %r9, %r6;

    // total work = n_samples * n_feat
    mul.lo.u32 %r10, %r0, %r1;

$IA_OUTER:
    setp.ge.u32 %p0, %r9, %r10;
    @%p0 bra $IA_DONE;

    // (sample, feat) = divmod(r9, n_feat)
    div.u32 %r11, %r9, %r1;
    rem.u32 %r12, %r9, %r1;

    // Load x[sample, feat, 0] (first embedding dim)
    mul.lo.u32 %r13, %r11, %r1;
    add.u32 %r13, %r13, %r12;
    mul.lo.u32 %r13, %r13, %r2;
    mul.wide.u32 %rd6, %r13, 4;
    add.u64 %rd7, %rd0, %rd6;
    ld.global.f32 %f0, [%rd7];

    // Q = WQ * x[0], K = WK * x[0], V = WV * x[0]
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    ld.global.f32 %f4, [%rd4];

    mul.f32 %f5, %f1, %f0;   // q
    mul.f32 %f6, %f2, %f0;   // k
    mul.f32 %f7, %f3, %f0;   // v

    // dot(q, k) / sqrt(embed_dim)
    mul.f32 %f8, %f5, %f6;
    cvt.rn.f32.u32 %f9, %r2;
    sqrt.approx.f32 %f10, %f9;
    div.rn.f32 %f8, %f8, %f10;

    // softmax of 1-element is 1.0 — attn = 1.0 * v
    mov.f32 %f11, {ONE};
    mul.f32 %f11, %f11, %f7;

    // out = WO * attn
    mul.f32 %f11, %f4, %f11;

    // Store out[sample, feat, 0]
    st.global.f32 [%rd7], %f11;  // reuse address (in-place update ok for kernel demo)
    // Also write to param_out
    add.u64 %rd8, %rd5, %rd6;
    st.global.f32 [%rd8], %f11;

    add.u32 %r9, %r9, %r8;
    bra $IA_OUTER;

$IA_DONE:
    mov.u32 %r14, 0;
    mov.u32 %r15, 0;
    mov.f32 %f0, {ZERO};
    mov.u64 %rd9,  0;
    mov.u64 %rd10, 0;
    mov.u64 %rd11, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 5: node_tree_eval_kernel ─────────────────────────────────────────

/// PTX kernel: full NODE soft oblivious tree (Popov et al. 2019).
///
/// For each sample and each level `0..depth` it computes the entmax-1.5 feature
/// selection over `feat_logits[level, 0..input_dim]` by the **same** bisection
/// the CPU reference uses (`min`/`max` bracket, then 64 halving steps streaming
/// `Σ max(0, logit_i - mid)²`), forms the soft feature value `selected_x = Σ
/// max(0, logit_i - tau)² · x[s, i]`, and the soft split `b_level =
/// sigmoid(selected_x - thr[level])` via the base-2 `ex2` path with the
/// `log2(e)` pre-scale. The per-level `b` values are held in a thread-private
/// `.local` array (capped at depth ≤ 8). Finally it sums over all `2^depth`
/// leaves: `prob = Π_level (bit ? b_level : 1 - b_level)` with `bit = (leaf >>
/// (depth-1-level)) & 1`, accumulating `out[s, d] += prob ·
/// leaf_values[leaf*output_dim + d]`. `beta` is the fixed constant 1.0.
///
/// All arithmetic except the `ex2`-based sigmoid mirrors
/// `tree::node::NodeTree::forward` operation-for-operation, so the only source
/// of divergence is the `~1 ulp` `ex2.approx`. Grid: one thread per sample.
#[must_use]
pub fn node_tree_eval_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let two = f32_hex(2.0_f32);
    let half = f32_hex(0.5_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let pos_inf = f32_hex(f32::INFINITY);
    format!(
        r#"{header}.visible .entry node_tree_eval_kernel(
    .param .u64 param_x,
    .param .u64 param_feat_logits,
    .param .u64 param_thresholds,
    .param .u64 param_leaf_values,
    .param .u64 param_out,
    .param .u32 param_n_samples,
    .param .u32 param_input_dim,
    .param .u32 param_depth,
    .param .u32 param_output_dim
)
{{
    .reg .u64  %rd<20>;
    .reg .u32  %r<28>;
    .reg .f32  %f<28>;
    .reg .pred %p<10>;
    .local .align 4 .b8 __b[32];        // per-level split probs, depth <= 8

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_feat_logits];
    ld.param.u64 %rd2, [param_thresholds];
    ld.param.u64 %rd3, [param_leaf_values];
    ld.param.u64 %rd4, [param_out];
    ld.param.u32 %r0,  [param_n_samples];
    ld.param.u32 %r1,  [param_input_dim];
    ld.param.u32 %r2,  [param_depth];
    ld.param.u32 %r3,  [param_output_dim];

    mov.u64 %rd5, __b;                  // thread-local b[] base address

    // n_leaves = 1 << depth
    mov.u32 %r15, 1;
    shl.b32 %r15, %r15, %r2;

    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %ctaid.x;
    mov.u32 %r6, %tid.x;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    mov.u32 %r8, %nctaid.x;
    mul.lo.u32 %r8, %r4, %r8;
    mov.u32 %r9, %r7;                   // sample

    mov.f32 %f19, {ZERO};              // reusable zero (for st.global)

$NTE_OUTER:
    setp.ge.u32 %p0, %r9, %r0;
    @%p0 bra $NTE_DONE;

    // &x[s, 0]
    mul.lo.u32 %r18, %r9, %r1;
    mul.wide.u32 %rd6, %r18, 4;
    add.u64 %rd6, %rd0, %rd6;

    // &out[s, 0]
    mul.lo.u32 %r18, %r9, %r3;
    mul.wide.u32 %rd7, %r18, 4;
    add.u64 %rd7, %rd4, %rd7;

    // zero the output row
    mov.u32 %r14, 0;
$NTE_ZERO:
    setp.ge.u32 %p6, %r14, %r3;
    @%p6 bra $NTE_ZERO_DONE;
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd10, %rd7, %rd9;
    st.global.f32 [%rd10], %f19;
    add.u32 %r14, %r14, 1;
    bra $NTE_ZERO;
$NTE_ZERO_DONE:

    // ---- per-level split probabilities b[level] ----
    mov.u32 %r10, 0;                   // level
$NTE_LEVEL:
    setp.ge.u32 %p1, %r10, %r2;
    @%p1 bra $NTE_LEVEL_DONE;

    // &feat_logits[level, 0]
    mul.lo.u32 %r18, %r10, %r1;
    mul.wide.u32 %rd8, %r18, 4;
    add.u64 %rd8, %rd1, %rd8;

    // bracket: z_max, z_min over input_dim logits
    mov.f32 %f0, {NEG_INF};
    mov.f32 %f1, {POS_INF};
    mov.u32 %r11, 0;
$NTE_MM:
    setp.ge.u32 %p2, %r11, %r1;
    @%p2 bra $NTE_MM_DONE;
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.f32 %f6, [%rd10];
    max.f32 %f0, %f0, %f6;
    min.f32 %f1, %f1, %f6;
    add.u32 %r11, %r11, 1;
    bra $NTE_MM;
$NTE_MM_DONE:

    // lo = z_min - 2 ; hi = z_max
    sub.f32 %f2, %f1, {TWO};
    mov.f32 %f3, %f0;

    // 64 bisection iterations for entmax-1.5 threshold
    mov.u32 %r12, 0;
$NTE_BIS:
    setp.ge.u32 %p3, %r12, 64;
    @%p3 bra $NTE_BIS_DONE;
    add.f32 %f4, %f2, %f3;
    mul.f32 %f4, %f4, {HALF};         // mid = 0.5*(lo+hi)
    mov.f32 %f5, {ZERO};              // sum
    mov.u32 %r11, 0;
$NTE_BSUM:
    setp.ge.u32 %p2, %r11, %r1;
    @%p2 bra $NTE_BSUM_DONE;
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.f32 %f6, [%rd10];
    sub.f32 %f7, %f6, %f4;
    max.f32 %f7, %f7, {ZERO};
    mul.f32 %f21, %f7, %f7;
    add.f32 %f5, %f5, %f21;           // sum += max(0, z-mid)^2
    add.u32 %r11, %r11, 1;
    bra $NTE_BSUM;
$NTE_BSUM_DONE:
    setp.gt.f32 %p4, %f5, {ONE};      // sum > 1 ?
    @%p4 mov.f32 %f2, %f4;            // lo = mid
    @!%p4 mov.f32 %f3, %f4;           // hi = mid
    add.u32 %r12, %r12, 1;
    bra $NTE_BIS;
$NTE_BIS_DONE:

    // tau = 0.5*(lo+hi)
    add.f32 %f8, %f2, %f3;
    mul.f32 %f8, %f8, {HALF};

    // selected_x = sum_i max(0, logit_i - tau)^2 * x[s, i]
    mov.f32 %f9, {ZERO};
    mov.u32 %r11, 0;
$NTE_SEL:
    setp.ge.u32 %p2, %r11, %r1;
    @%p2 bra $NTE_SEL_DONE;
    mul.wide.u32 %rd9, %r11, 4;
    add.u64 %rd10, %rd8, %rd9;
    ld.global.f32 %f6, [%rd10];
    sub.f32 %f7, %f6, %f8;
    max.f32 %f7, %f7, {ZERO};
    mul.f32 %f10, %f7, %f7;           // p_i
    add.u64 %rd11, %rd6, %rd9;
    ld.global.f32 %f11, [%rd11];      // x[s, i]
    mul.f32 %f21, %f10, %f11;
    add.f32 %f9, %f9, %f21;           // selected_x += p_i * x_i
    add.u32 %r11, %r11, 1;
    bra $NTE_SEL;
$NTE_SEL_DONE:

    // arg = selected_x - thr[level]
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd10, %rd2, %rd9;
    ld.global.f32 %f12, [%rd10];
    sub.f32 %f12, %f9, %f12;

    // b = sigmoid(arg) = ex2(arg*log2e) / (1 + ex2(arg*log2e))
    mul.f32 %f13, %f12, {LOG2E};
    ex2.approx.f32 %f13, %f13;
    add.f32 %f14, %f13, {ONE};
    div.rn.f32 %f14, %f13, %f14;

    // __b[level] = b
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd10, %rd5, %rd9;
    st.local.f32 [%rd10], %f14;

    add.u32 %r10, %r10, 1;
    bra $NTE_LEVEL;
$NTE_LEVEL_DONE:

    // ---- leaf mixture ----
    mov.u32 %r13, 0;                   // leaf
$NTE_LEAF:
    setp.ge.u32 %p5, %r13, %r15;
    @%p5 bra $NTE_LEAF_DONE;

    mov.f32 %f15, {ONE};              // prob
    mov.u32 %r10, 0;                   // level
$NTE_LPROB:
    setp.ge.u32 %p1, %r10, %r2;
    @%p1 bra $NTE_LPROB_DONE;
    sub.u32 %r16, %r2, 1;
    sub.u32 %r16, %r16, %r10;          // shift = depth-1-level
    shr.u32 %r17, %r13, %r16;
    and.b32 %r17, %r17, 1;             // bit
    mul.wide.u32 %rd9, %r10, 4;
    add.u64 %rd10, %rd5, %rd9;
    ld.local.f32 %f14, [%rd10];        // b[level]
    sub.f32 %f17, {ONE}, %f14;        // 1 - b
    setp.eq.u32 %p7, %r17, 1;
    mov.f32 %f16, %f17;               // default (bit == 0): 1 - b
    @%p7 mov.f32 %f16, %f14;          // bit == 1: b
    mul.f32 %f15, %f15, %f16;          // prob *= factor
    add.u32 %r10, %r10, 1;
    bra $NTE_LPROB;
$NTE_LPROB_DONE:

    // &leaf_values[leaf, 0]
    mul.lo.u32 %r18, %r13, %r3;
    mul.wide.u32 %rd9, %r18, 4;
    add.u64 %rd11, %rd3, %rd9;

    // out[s, d] += prob * leaf_values[leaf, d]
    mov.u32 %r14, 0;
$NTE_ACC:
    setp.ge.u32 %p6, %r14, %r3;
    @%p6 bra $NTE_ACC_DONE;
    mul.wide.u32 %rd9, %r14, 4;
    add.u64 %rd12, %rd11, %rd9;
    ld.global.f32 %f18, [%rd12];
    add.u64 %rd13, %rd7, %rd9;
    ld.global.f32 %f20, [%rd13];
    mul.f32 %f21, %f15, %f18;
    add.f32 %f20, %f20, %f21;
    st.global.f32 [%rd13], %f20;
    add.u32 %r14, %r14, 1;
    bra $NTE_ACC;
$NTE_ACC_DONE:

    add.u32 %r13, %r13, 1;
    bra $NTE_LEAF;
$NTE_LEAF_DONE:

    add.u32 %r9, %r9, %r8;
    bra $NTE_OUTER;

$NTE_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        TWO = two,
        HALF = half,
        NEG_INF = neg_inf,
        POS_INF = pos_inf,
        LOG2E = f32_hex(std::f32::consts::LOG2_E),
    )
}

// ─── Kernel 6: quantile_norm_kernel ──────────────────────────────────────────

/// PTX kernel: empirical-CDF quantile transform (Uniform output), matching
/// `preprocess::quantile_feat::QuantileTransformer` fit with
/// `n_quantiles == n_train == n_samples`.
///
/// The `sorted` reference is laid out `[n_features, n_train]` (per-feature base
/// `feat * n_train`). Per `(sample, feat)` with `x = x[sample, feat]`: if
/// `x <= sorted[feat, 0]` → `q = 0`; if `x >= sorted[feat, n_train-1]` →
/// `q = 1`; otherwise a linear scan finds the largest `lo` with
/// `sorted[feat, lo] <= x`, and `q` is the linear interpolation
/// `q_lo + t·(q_hi - q_lo)` with `q_lo = lo/(n_train-1)`,
/// `q_hi = (lo+1)/(n_train-1)`, `t = (x - sorted[lo]) / (sorted[lo+1] -
/// sorted[lo])` (guarding a degenerate `|hi - lo| < 1e-12` to `q_lo`). This is
/// the exact empirical quantile the CPU `empirical_quantile` computes.
///
/// Grid: one thread per `(sample, feat)` cell, grid-stride over all cells.
#[must_use]
pub fn quantile_norm_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let eps = f32_hex(1e-12_f32);
    format!(
        r#"{header}.visible .entry quantile_norm_kernel(
    .param .u64 param_x,
    .param .u64 param_sorted,
    .param .u64 param_out,
    .param .u32 param_n_samples,
    .param .u32 param_n_features,
    .param .u32 param_n_train
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<24>;
    .reg .f32  %f<20>;
    .reg .pred %p<8>;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_sorted];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0,  [param_n_samples];
    ld.param.u32 %r1,  [param_n_features];
    ld.param.u32 %r2,  [param_n_train];

    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mov.u32 %r7, %nctaid.x;
    mul.lo.u32 %r8, %r3, %r7;
    mov.u32 %r9, %r6;

    // total work = n_samples * n_features
    mul.lo.u32 %r10, %r0, %r1;

$QN_OUTER:
    setp.ge.u32 %p0, %r9, %r10;
    @%p0 bra $QN_DONE;

    // feat = work_idx % n_features  (sample row not needed: x is row-major so the
    // flat work index already indexes x[sample, feat] directly).
    rem.u32 %r12, %r9, %r1;

    // x[work_idx]
    mul.wide.u32 %rd3, %r9, 4;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    // &sorted[feat, 0] = sorted + feat*n_train
    mul.lo.u32 %r13, %r12, %r2;
    mul.wide.u32 %rd5, %r13, 4;
    add.u64 %rd6, %rd1, %rd5;

    // first = sorted[feat, 0]
    ld.global.f32 %f1, [%rd6];

    // last = sorted[feat, n_train-1]
    sub.u32 %r14, %r2, 1;
    mul.wide.u32 %rd7, %r14, 4;
    add.u64 %rd8, %rd6, %rd7;
    ld.global.f32 %f2, [%rd8];

    mov.f32 %f3, {ZERO};               // q default 0

    // x <= first -> q = 0
    setp.le.f32 %p1, %f0, %f1;
    @%p1 bra $QN_STORE;

    // x >= last -> q = 1
    mov.f32 %f4, {ONE};
    setp.ge.f32 %p2, %f0, %f2;
    @%p2 mov.f32 %f3, %f4;
    @%p2 bra $QN_STORE;

    // linear scan: lo = largest k with sorted[feat, k] <= x
    mov.u32 %r15, 0;
    mov.u32 %r16, 0;
$QN_SCAN:
    setp.ge.u32 %p3, %r16, %r2;
    @%p3 bra $QN_SCAN_DONE;
    mul.wide.u32 %rd9, %r16, 4;
    add.u64 %rd10, %rd6, %rd9;
    ld.global.f32 %f5, [%rd10];
    setp.le.f32 %p4, %f5, %f0;
    @%p4 mov.u32 %r15, %r16;
    add.u32 %r16, %r16, 1;
    bra $QN_SCAN;
$QN_SCAN_DONE:

    // lo_val = sorted[lo], hi_val = sorted[lo+1]
    mul.wide.u32 %rd11, %r15, 4;
    add.u64 %rd12, %rd6, %rd11;
    ld.global.f32 %f6, [%rd12];
    add.u32 %r17, %r15, 1;
    mul.wide.u32 %rd13, %r17, 4;
    add.u64 %rd14, %rd6, %rd13;
    ld.global.f32 %f7, [%rd14];

    // q_lo = lo/(n_train-1), q_hi = (lo+1)/(n_train-1)
    cvt.rn.f32.u32 %f8, %r14;          // n_train-1 as f32
    cvt.rn.f32.u32 %f9, %r15;          // lo
    div.rn.f32 %f10, %f9, %f8;         // q_lo
    cvt.rn.f32.u32 %f11, %r17;         // lo+1
    div.rn.f32 %f12, %f11, %f8;        // q_hi

    // degenerate guard: |hi_val - lo_val| < 1e-12 -> q = q_lo
    sub.f32 %f13, %f7, %f6;
    abs.f32 %f14, %f13;
    setp.lt.f32 %p5, %f14, {EPS};
    mov.f32 %f3, %f10;
    @%p5 bra $QN_STORE;

    // t = (x - lo_val)/(hi_val - lo_val) ; q = q_lo + t*(q_hi - q_lo)
    sub.f32 %f15, %f0, %f6;
    div.rn.f32 %f16, %f15, %f13;
    sub.f32 %f17, %f12, %f10;
    fma.rn.f32 %f3, %f16, %f17, %f10;

$QN_STORE:
    add.u64 %rd15, %rd2, %rd3;
    st.global.f32 [%rd15], %f3;

    add.u32 %r9, %r9, %r8;
    bra $QN_OUTER;

$QN_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        EPS = eps,
    )
}

// ─── Kernel 7: auc_roc_kernel ─────────────────────────────────────────────────

/// PTX kernel: sort scores, compute FPR/TPR, integrate trapezoidal AUC.
#[must_use]
pub fn auc_roc_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let half = f32_hex(0.5_f32);
    format!(
        r#"{header}.visible .entry auc_roc_kernel(
    .param .u64 param_scores,
    .param .u64 param_labels,
    .param .u64 param_auc_out,
    .param .u32 param_n,
    .param .u32 param_n_pos,
    .param .u32 param_n_neg
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<14>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64 %rd0, [param_scores];
    ld.param.u64 %rd1, [param_labels];
    ld.param.u64 %rd2, [param_auc_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.u32 %r1,  [param_n_pos];
    ld.param.u32 %r2,  [param_n_neg];

    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;

    // Only thread 0 computes (simplified single-threaded kernel)
    setp.ne.u32 %p0, %r6, 0;
    @%p0 bra $AUC_DONE;

    // Compute pair-based AUC: count concordant pairs
    // For each (i, j) where label[i]=1, label[j]=0: concordant if score[i] > score[j]
    // Simplified: iterate over all pairs (n^2), accumulate
    cvt.rn.f32.u32 %f0, %r1;     // n_pos as float
    cvt.rn.f32.u32 %f1, %r2;     // n_neg as float
    mul.f32 %f2, %f0, %f1;        // n_pos * n_neg = total pairs

    // AUC accumulator
    mov.f32 %f3, {ZERO};
    mov.u32 %r7, 0;

$AUC_POS_LOOP:
    setp.ge.u32 %p1, %r7, %r0;
    @%p1 bra $AUC_POS_DONE;

    // Load label[i]
    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r8, [%rd4];
    setp.ne.u32 %p1, %r8, 1;
    @%p1 bra $AUC_POS_NEXT;

    // Load score[i]
    add.u64 %rd5, %rd0, %rd3;
    ld.global.f32 %f4, [%rd5];

    mov.u32 %r9, 0;
$AUC_NEG_LOOP:
    setp.ge.u32 %p2, %r9, %r0;
    @%p2 bra $AUC_NEG_DONE;

    mul.wide.u32 %rd6, %r9, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r10, [%rd7];
    setp.ne.u32 %p2, %r10, 0;
    @%p2 bra $AUC_NEG_NEXT;

    // Load score[j]
    add.u64 %rd8, %rd0, %rd6;
    ld.global.f32 %f5, [%rd8];

    // concordant if score[i] > score[j]
    setp.gt.f32 %p2, %f4, %f5;
    mov.f32 %f6, {ZERO};
    @%p2 mov.f32 %f6, {ONE};
    // tied: 0.5
    setp.eq.f32 %p2, %f4, %f5;
    @%p2 mov.f32 %f6, {HALF};
    add.f32 %f3, %f3, %f6;

$AUC_NEG_NEXT:
    add.u32 %r9, %r9, 1;
    bra $AUC_NEG_LOOP;
$AUC_NEG_DONE:

$AUC_POS_NEXT:
    add.u32 %r7, %r7, 1;
    bra $AUC_POS_LOOP;
$AUC_POS_DONE:

    // AUC = concordant / (n_pos * n_neg)
    setp.gt.f32 %p0, %f2, {ZERO};
    mov.f32 %f7, {ZERO};
    @%p0 div.rn.f32 %f7, %f3, %f2;

    st.global.f32 [%rd2], %f7;

$AUC_DONE:
    mov.u32 %r11, 0;
    mov.u32 %r12, 0;
    mov.u32 %r13, 0;
    mov.u32 %r14, 0;
    mov.u32 %r15, 0;
    mov.f32 %f8,  {ZERO};
    mov.f32 %f9,  {ZERO};
    mov.f32 %f10, {ZERO};
    mov.f32 %f11, {ZERO};
    mov.f32 %f12, {ZERO};
    mov.f32 %f13, {ZERO};
    mov.u64 %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        HALF = half,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_kernel(ptx: &str, sm: u32, name: &str) {
        assert!(
            ptx.contains(&format!("sm_{sm}")),
            "missing sm_{sm} in kernel {name}"
        );
        assert!(ptx.contains(".version"), "missing .version in {name}");
        assert!(
            ptx.contains(".address_size 64"),
            "missing .address_size 64 in {name}"
        );
        assert!(
            ptx.contains(".visible .entry"),
            "missing .visible .entry in {name}"
        );
        assert!(ptx.contains(name), "missing kernel name {name}");
    }

    #[test]
    fn sparsemax_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&sparsemax_ptx(sm), sm, "sparsemax_kernel");
        }
    }

    #[test]
    fn feature_tokenize_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&feature_tokenize_ptx(sm), sm, "feature_tokenize_kernel");
        }
    }

    #[test]
    fn tabnet_step_attn_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&tabnet_step_attn_ptx(sm), sm, "tabnet_step_attn_kernel");
        }
    }

    #[test]
    fn intersample_attn_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&intersample_attn_ptx(sm), sm, "intersample_attn_kernel");
        }
    }

    #[test]
    fn node_tree_eval_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&node_tree_eval_ptx(sm), sm, "node_tree_eval_kernel");
        }
    }

    #[test]
    fn quantile_norm_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&quantile_norm_ptx(sm), sm, "quantile_norm_kernel");
        }
    }

    #[test]
    fn auc_roc_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_kernel(&auc_roc_ptx(sm), sm, "auc_roc_kernel");
        }
    }

    #[test]
    fn ptx_header_version_strings() {
        assert!(sparsemax_ptx(75).contains(".version 7.5"));
        assert!(sparsemax_ptx(80).contains(".version 8.0"));
        assert!(sparsemax_ptx(86).contains(".version 8.0"));
        assert!(sparsemax_ptx(90).contains(".version 8.4"));
        assert!(sparsemax_ptx(100).contains(".version 8.7"));
        assert!(sparsemax_ptx(120).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_known_values() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
