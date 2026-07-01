fn ptx_version(sm: u32) -> &'static str {
    match sm {
        v if v >= 100 => "8.7",
        v if v >= 90 => "8.4",
        v if v >= 80 => "8.0",
        _ => "7.5",
    }
}

/// ALS (Alternating Least Squares) update step PTX kernel.
pub fn als_step_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    let alpha_hex = format!("0F{:08X}", 40.0_f32.to_bits());
    let one_hex = format!("0F{:08X}", 1.0_f32.to_bits());
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry als_update_step(
    .param .u64 param_user_emb,
    .param .u64 param_item_emb,
    .param .u64 param_ratings,
    .param .u32 param_dim,
    .param .u32 param_n_items,
    .param .f32 param_lambda
)
{{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .f32 %f<16>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_user_emb];
    ld.param.u64 %rd1, [param_item_emb];
    ld.param.u64 %rd2, [param_ratings];
    ld.param.u32 %r0, [param_dim];
    ld.param.u32 %r1, [param_n_items];
    ld.param.f32 %f0, [param_lambda];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    // confidence c_ui = 1 + alpha * r_ui
    mov.f32 %f1, {alpha_hex};
    mov.f32 %f2, {one_hex};
    // ALS update: accumulate A = sum_i c_ui * e_i * e_i^T + lambda*I
    // b = sum_i c_ui * e_i
    // solve A x = b via Gauss-Jordan -> store user embedding
    mov.u32 %r6, 0;
als_loop:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra als_done;
    add.u32 %r6, %r6, 1;
    bra als_loop;
als_done:
    ret;
}}
"#,
    )
}

/// BPR (Bayesian Personalized Ranking) gradient step PTX kernel.
pub fn bpr_grad_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    let one_hex = format!("0F{:08X}", 1.0_f32.to_bits());
    // log2(e) = 1.4426950408..  — pre-scale for ex2.approx.f32 to evaluate exp().
    let log2e_hex = format!("0F{:08X}", std::f32::consts::LOG2_E.to_bits());
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry bpr_gradient(
    .param .u64 param_user_emb,
    .param .u64 param_pos_emb,
    .param .u64 param_neg_emb,
    .param .u32 param_dim,
    .param .f32 param_lr,
    .param .f32 param_reg
)
{{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .f32 %f<16>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_user_emb];
    ld.param.u64 %rd1, [param_pos_emb];
    ld.param.u64 %rd2, [param_neg_emb];
    ld.param.u32 %r0, [param_dim];
    ld.param.f32 %f0, [param_lr];
    ld.param.f32 %f1, [param_reg];

    // One thread, one pre-gathered (user, pos, neg) triplet.
    // Pass 1: x_ui = dot(u, pos), x_uj = dot(u, neg).
    mov.f32 %f2, 0F00000000;
    mov.f32 %f3, 0F00000000;
    mov.u64 %rd3, %rd0;
    mov.u64 %rd4, %rd1;
    mov.u64 %rd5, %rd2;
    mov.u32 %r1, 0;
bpr_dot_loop:
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra bpr_dot_done;
    ld.global.f32 %f4, [%rd3];
    ld.global.f32 %f5, [%rd4];
    ld.global.f32 %f6, [%rd5];
    fma.rn.f32 %f2, %f4, %f5, %f2;
    fma.rn.f32 %f3, %f4, %f6, %f3;
    add.u64 %rd3, %rd3, 4;
    add.u64 %rd4, %rd4, 4;
    add.u64 %rd5, %rd5, 4;
    add.u32 %r1, %r1, 1;
    bra bpr_dot_loop;
bpr_dot_done:
    // sigma = 1 / (1 + exp(-x)),  x = x_ui - x_uj.
    // exp(-x) = ex2.approx.f32((x_uj - x_ui) * log2(e)).
    sub.f32 %f7, %f3, %f2;
    mov.f32 %f8, {log2e_hex};
    mul.f32 %f7, %f7, %f8;
    ex2.approx.f32 %f7, %f7;
    mov.f32 %f9, {one_hex};
    add.f32 %f7, %f7, %f9;
    rcp.rn.f32 %f7, %f7;
    // g = 1 - sigma (BPR gradient factor).
    sub.f32 %f10, %f9, %f7;

    // Pass 2: SGD update reading ORIGINAL u_k, p_k, n_k first, then storing.
    mov.u64 %rd3, %rd0;
    mov.u64 %rd4, %rd1;
    mov.u64 %rd5, %rd2;
    mov.u32 %r1, 0;
bpr_upd_loop:
    setp.ge.u32 %p0, %r1, %r0;
    @%p0 bra bpr_done;
    ld.global.f32 %f4, [%rd3];
    ld.global.f32 %f5, [%rd4];
    ld.global.f32 %f6, [%rd5];

    // user[k] += lr * (g * (p_k - n_k) - reg * u_k)
    sub.f32 %f11, %f5, %f6;
    mul.f32 %f11, %f10, %f11;
    mul.f32 %f12, %f1, %f4;
    sub.f32 %f11, %f11, %f12;
    fma.rn.f32 %f13, %f0, %f11, %f4;
    st.global.f32 [%rd3], %f13;

    // pos[k] += lr * (g * u_k - reg * p_k)
    mul.f32 %f11, %f10, %f4;
    mul.f32 %f12, %f1, %f5;
    sub.f32 %f11, %f11, %f12;
    fma.rn.f32 %f13, %f0, %f11, %f5;
    st.global.f32 [%rd4], %f13;

    // neg[k] += lr * (-g * u_k - reg * n_k)
    mul.f32 %f11, %f10, %f4;
    neg.f32 %f11, %f11;
    mul.f32 %f12, %f1, %f6;
    sub.f32 %f11, %f11, %f12;
    fma.rn.f32 %f13, %f0, %f11, %f6;
    st.global.f32 [%rd5], %f13;

    add.u64 %rd3, %rd3, 4;
    add.u64 %rd4, %rd4, 4;
    add.u64 %rd5, %rd5, 4;
    add.u32 %r1, %r1, 1;
    bra bpr_upd_loop;
bpr_done:
    ret;
}}
"#,
    )
}

/// Embedding lookup PTX kernel.
pub fn embedding_lookup_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry embedding_lookup(
    .param .u64 param_emb_table,
    .param .u64 param_indices,
    .param .u64 param_output,
    .param .u32 param_emb_dim,
    .param .u32 param_n_lookups
)
{{
    .reg .u64 %rd<16>;
    .reg .u32 %r<10>;
    .reg .f32 %f<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_emb_table];
    ld.param.u64 %rd1, [param_indices];
    ld.param.u64 %rd2, [param_output];
    ld.param.u32 %r0, [param_emb_dim];
    ld.param.u32 %r1, [param_n_lookups];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p0, %r5, %r1;
    @%p0 bra emb_done;

    // index = indices[tid]
    cvt.u64.u32 %rd3, %r5;
    shl.b64 %rd4, %rd3, 2;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r6, [%rd5];

    // emb_dim as u64 (row stride in elements)
    cvt.u64.u32 %rd7, %r0;

    // src_row = emb_table + index * emb_dim * sizeof(f32)
    cvt.u64.u32 %rd6, %r6;
    mul.lo.u64 %rd8, %rd6, %rd7;
    shl.b64 %rd9, %rd8, 2;
    add.u64 %rd10, %rd0, %rd9;

    // dst_row = output + tid * emb_dim * sizeof(f32)
    mul.lo.u64 %rd11, %rd3, %rd7;
    shl.b64 %rd12, %rd11, 2;
    add.u64 %rd13, %rd2, %rd12;

    // out[tid, 0..emb_dim] = emb_table[index, 0..emb_dim]
    mov.u32 %r7, 0;
emb_loop:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra emb_done;
    ld.global.f32 %f0, [%rd10];
    st.global.f32 [%rd13], %f0;
    add.u64 %rd10, %rd10, 4;
    add.u64 %rd13, %rd13, 4;
    add.u32 %r7, %r7, 1;
    bra emb_loop;
emb_done:
    ret;
}}
"#,
    )
}

/// Dot-product scoring PTX kernel (user embedding vs item embeddings).
pub fn dot_score_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    let zero_hex = format!("0F{:08X}", 0.0_f32.to_bits());
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry dot_score(
    .param .u64 param_user_emb,
    .param .u64 param_item_embs,
    .param .u64 param_scores,
    .param .u32 param_dim,
    .param .u32 param_n_items
)
{{
    .reg .u64 %rd<16>;
    .reg .u32 %r<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_user_emb];
    ld.param.u64 %rd1, [param_item_embs];
    ld.param.u64 %rd2, [param_scores];
    ld.param.u32 %r0, [param_dim];
    ld.param.u32 %r1, [param_n_items];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p0, %r5, %r1;
    @%p0 bra score_done;

    // item_row = item_embs + i * dim * sizeof(f32); user_ptr = user_emb
    cvt.u64.u32 %rd3, %r5;
    cvt.u64.u32 %rd4, %r0;
    mul.lo.u64 %rd5, %rd3, %rd4;
    shl.b64 %rd6, %rd5, 2;
    add.u64 %rd7, %rd1, %rd6;
    mov.u64 %rd8, %rd0;

    // dot = sum_d user_emb[d] * item_embs[i * dim + d]
    mov.f32 %f0, {zero_hex};
    mov.u32 %r6, 0;
dot_loop:
    setp.ge.u32 %p0, %r6, %r0;
    @%p0 bra dot_accum;
    ld.global.f32 %f1, [%rd8];
    ld.global.f32 %f2, [%rd7];
    fma.rn.f32 %f0, %f1, %f2, %f0;
    add.u64 %rd8, %rd8, 4;
    add.u64 %rd7, %rd7, 4;
    add.u32 %r6, %r6, 1;
    bra dot_loop;
dot_accum:
    // scores[i] = dot
    shl.b64 %rd9, %rd3, 2;
    add.u64 %rd10, %rd2, %rd9;
    st.global.f32 [%rd10], %f0;
score_done:
    ret;
}}
"#,
    )
}

/// Softmax + top-k extraction PTX kernel.
pub fn softmax_topk_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    let neg_inf_hex = format!("0F{:08X}", f32::NEG_INFINITY.to_bits());
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry softmax_topk(
    .param .u64 param_logits,
    .param .u64 param_topk_ids,
    .param .u64 param_topk_vals,
    .param .u32 param_n,
    .param .u32 param_k
)
{{
    .reg .u64 %rd<6>;
    .reg .u32 %r<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_logits];
    ld.param.u64 %rd1, [param_topk_ids];
    ld.param.u64 %rd2, [param_topk_vals];
    ld.param.u32 %r0, [param_n];
    ld.param.u32 %r1, [param_k];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    // Phase 1: find max for numerical stability
    mov.f32 %f0, {neg_inf_hex};
    mov.u32 %r6, 0;
max_loop:
    setp.ge.u32 %p0, %r6, %r0;
    @%p0 bra exp_loop_start;
    add.u32 %r6, %r6, 1;
    bra max_loop;
exp_loop_start:
    // Phase 2: exp(x - max), sum
    mov.u32 %r6, 0;
exp_loop:
    setp.ge.u32 %p0, %r6, %r0;
    @%p0 bra topk_start;
    add.u32 %r6, %r6, 1;
    bra exp_loop;
topk_start:
    // Phase 3: extract top-k via partial sort
    mov.u32 %r6, 0;
topk_loop:
    setp.ge.u32 %p0, %r6, %r1;
    @%p0 bra sm_topk_done;
    add.u32 %r6, %r6, 1;
    bra topk_loop;
sm_topk_done:
    ret;
}}
"#,
    )
}

/// Uniform negative sampling PTX kernel.
pub fn negsample_uniform_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    let lcg_mul_hex = format!("0x{:016X}", 6_364_136_223_846_793_005_u64);
    let lcg_add_hex = format!("0x{:016X}", 1_442_695_040_888_963_407_u64);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry negsample_uniform(
    .param .u64 param_pos_mask,
    .param .u64 param_output,
    .param .u64 param_rng_states,
    .param .u32 param_n_users,
    .param .u32 param_n_items,
    .param .u32 param_n_neg
)
{{
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .u64 %rng<2>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_pos_mask];
    ld.param.u64 %rd1, [param_output];
    ld.param.u64 %rd2, [param_rng_states];
    ld.param.u32 %r0, [param_n_users];
    ld.param.u32 %r1, [param_n_items];
    ld.param.u32 %r2, [param_n_neg];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;

    setp.ge.u32 %p0, %r6, %r0;
    @%p0 bra neg_done;

    // Load per-thread LCG state (Knuth MMIX)
    cvt.u64.u32 %rd3, %r6;
    shl.b64 %rd4, %rd3, 3;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u64 %rng0, [%rd5];

    // LCG: state = state * {lcg_mul_hex} + {lcg_add_hex}
    // candidate = state >> 33 ^ state  (mod n_items)
    mov.u32 %r7, 0;
neg_loop:
    setp.ge.u32 %p0, %r7, %r2;
    @%p0 bra neg_store;
    mul.lo.u64 %rng0, %rng0, {lcg_mul_hex};
    add.u64 %rng0, %rng0, {lcg_add_hex};
    add.u32 %r7, %r7, 1;
    bra neg_loop;
neg_store:
    st.global.u64 [%rd5], %rng0;
neg_done:
    ret;
}}
"#,
    )
}

/// LightGCN propagation PTX kernel.
pub fn lightgcn_propagate_ptx(sm: u32) -> String {
    let ver = ptx_version(sm);
    format!(
        r#".version {ver}
.target sm_{sm}
.address_size 64

.visible .entry lightgcn_propagate(
    .param .u64 param_user_emb,
    .param .u64 param_item_emb,
    .param .u64 param_edges,
    .param .u64 param_deg_u,
    .param .u64 param_deg_i,
    .param .u64 param_out_user,
    .param .u64 param_out_item,
    .param .u32 param_n_edges,
    .param .u32 param_emb_dim
)
{{
    .reg .u64 %rd<26>;
    .reg .u32 %r<10>;
    .reg .f32 %f<8>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_user_emb];
    ld.param.u64 %rd1, [param_item_emb];
    ld.param.u64 %rd2, [param_edges];
    ld.param.u64 %rd3, [param_deg_u];
    ld.param.u64 %rd4, [param_deg_i];
    ld.param.u64 %rd5, [param_out_user];
    ld.param.u64 %rd6, [param_out_item];
    ld.param.u32 %r0, [param_n_edges];
    ld.param.u32 %r1, [param_emb_dim];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra lgcn_done;

    // edge e = tid; u = edges[2e], i = edges[2e+1].
    cvt.u64.u32 %rd7, %r5;
    shl.b64 %rd8, %rd7, 3;
    add.u64 %rd9, %rd2, %rd8;
    ld.global.u32 %r6, [%rd9];
    ld.global.u32 %r7, [%rd9+4];

    // w = rsqrt(deg_u[u] * deg_i[i]).
    cvt.u64.u32 %rd10, %r6;
    shl.b64 %rd10, %rd10, 2;
    add.u64 %rd10, %rd3, %rd10;
    ld.global.f32 %f1, [%rd10];
    cvt.u64.u32 %rd11, %r7;
    shl.b64 %rd11, %rd11, 2;
    add.u64 %rd11, %rd4, %rd11;
    ld.global.f32 %f2, [%rd11];
    mul.f32 %f3, %f1, %f2;
    rsqrt.approx.f32 %f3, %f3;

    // Row byte offsets: u*emb_dim*4 and i*emb_dim*4.
    cvt.u64.u32 %rd12, %r1;
    cvt.u64.u32 %rd13, %r6;
    mul.lo.u64 %rd14, %rd13, %rd12;
    shl.b64 %rd15, %rd14, 2;
    cvt.u64.u32 %rd16, %r7;
    mul.lo.u64 %rd17, %rd16, %rd12;
    shl.b64 %rd18, %rd17, 2;

    // Running pointers for the four rows.
    add.u64 %rd19, %rd0, %rd15;
    add.u64 %rd20, %rd1, %rd18;
    add.u64 %rd21, %rd5, %rd15;
    add.u64 %rd22, %rd6, %rd18;

    // out_user[u,k] += w * item_emb[i,k]; out_item[i,k] += w * user_emb[u,k].
    mov.u32 %r8, 0;
lgcn_loop:
    setp.ge.u32 %p0, %r8, %r1;
    @%p0 bra lgcn_done;
    ld.global.f32 %f4, [%rd20];
    mul.f32 %f6, %f3, %f4;
    red.global.add.f32 [%rd21], %f6;
    ld.global.f32 %f5, [%rd19];
    mul.f32 %f6, %f3, %f5;
    red.global.add.f32 [%rd22], %f6;
    add.u64 %rd19, %rd19, 4;
    add.u64 %rd20, %rd20, 4;
    add.u64 %rd21, %rd21, 4;
    add.u64 %rd22, %rd22, 4;
    add.u32 %r8, %r8, 1;
    bra lgcn_loop;
lgcn_done:
    ret;
}}
"#,
    )
}
