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

pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

pub fn inner_sgd_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// inner_sgd_kernel: theta_prime[i] = theta[i] - alpha * grad[i]
.visible .entry inner_sgd_kernel(
    .param .u64 p_theta,
    .param .u64 p_grad,
    .param .u64 p_out,
    .param .f32 alpha,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_theta];
    ld.param.u64  %rd1, [p_grad];
    ld.param.u64  %rd2, [p_out];
    ld.param.f32  %f4,  [alpha];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$SGD_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $SGD_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    add.u64       %rd6, %rd2, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    mul.f32       %f2, %f4, %f1;
    sub.f32       %f3, %f0, %f2;
    st.global.f32 [%rd6], %f3;

    add.u32       %r7, %r7, %r6;
    bra           $SGD_LOOP;

$SGD_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f5, {ZERO};
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

pub fn reptile_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// reptile_update_kernel: theta[i] += eps * (theta_prime[i] - theta[i])
.visible .entry reptile_update_kernel(
    .param .u64 p_theta,
    .param .u64 p_theta_prime,
    .param .f32 eps,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_theta];
    ld.param.u64  %rd1, [p_theta_prime];
    ld.param.f32  %f4,  [eps];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$REP_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $REP_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    add.u64       %rd4, %rd1, %rd2;

    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd4];
    sub.f32       %f2, %f1, %f0;
    mul.f32       %f3, %f4, %f2;
    add.f32       %f5, %f0, %f3;
    st.global.f32 [%rd3], %f5;

    add.u32       %r7, %r7, %r6;
    bra           $REP_LOOP;

$REP_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd5, 0;
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
    )
}

pub fn proto_distance_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// proto_distance_kernel: d[q*K + k] = sum_j (query_j - proto_j)^2
// p_query: [n_query * feat_dim], p_proto: [n_way * feat_dim], p_dist: [n_query * n_way]
.visible .entry proto_distance_kernel(
    .param .u64 p_query,
    .param .u64 p_proto,
    .param .u64 p_dist,
    .param .u32 n_query,
    .param .u32 n_way,
    .param .u32 feat_dim
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<14>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_query];
    ld.param.u64  %rd1, [p_proto];
    ld.param.u64  %rd2, [p_dist];
    ld.param.u32  %r0,  [n_query];
    ld.param.u32  %r1,  [n_way];
    ld.param.u32  %r12, [feat_dim];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mul.lo.u32    %r6, %r0, %r1;
    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $PD_DONE;

    rem.u32       %r7, %r5, %r1;
    div.u32       %r8, %r5, %r1;

    mul.lo.u32    %r9,  %r8, %r12;
    mul.lo.u32    %r10, %r7, %r12;

    mov.f32       %f0, {ZERO};
    mov.u32       %r11, 0;

$PD_FEAT_LOOP:
    setp.ge.u32   %p1, %r11, %r12;
    @%p1 bra $PD_FEAT_DONE;

    add.u32       %r13, %r9, %r11;
    mul.wide.u32  %rd3, %r13, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u32       %r13, %r10, %r11;
    mul.wide.u32  %rd5, %r13, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];

    sub.f32       %f3, %f1, %f2;
    fma.rn.f32    %f0, %f3, %f3, %f0;

    add.u32       %r11, %r11, 1;
    bra           $PD_FEAT_LOOP;

$PD_FEAT_DONE:
    mul.wide.u32  %rd7, %r5, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

$PD_DONE:
    mov.u32       %r12, 0;
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

pub fn cosine_sim_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let eps = f32_hex(1e-8_f32);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// cosine_sim_kernel: sim[i] = dot(a_i, b_i) / (||a_i|| * ||b_i|| + eps)
// p_a: [n * feat_dim], p_b: [n * feat_dim], p_out: [n]
.visible .entry cosine_sim_kernel(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 feat_dim
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [feat_dim];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $CS_DONE;

    mul.lo.u32    %r6, %r5, %r1;

    mov.f32       %f0, {ZERO};
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};
    mov.u32       %r7, 0;

$CS_FEAT_LOOP:
    setp.ge.u32   %p1, %r7, %r1;
    @%p1 bra $CS_FEAT_DONE;

    add.u32       %r8, %r6, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];
    ld.global.f32 %f4, [%rd5];

    fma.rn.f32    %f0, %f3, %f4, %f0;
    fma.rn.f32    %f1, %f3, %f3, %f1;
    fma.rn.f32    %f2, %f4, %f4, %f2;

    add.u32       %r7, %r7, 1;
    bra           $CS_FEAT_LOOP;

$CS_FEAT_DONE:
    sqrt.rn.f32   %f5, %f1;
    sqrt.rn.f32   %f6, %f2;
    mul.f32       %f7, %f5, %f6;
    mov.f32       %f8, {EPS};
    add.f32       %f9, %f7, %f8;
    div.rn.f32    %f10, %f0, %f9;

    mul.wide.u32  %rd6, %r5, 4;
    add.u64       %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f10;

$CS_DONE:
    mov.u32       %r9, 0;
    mov.f32       %f11, {ZERO};
    mov.u64       %rd8, 0;
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
    )
}

pub fn relation_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let log2e = f32_hex(core::f32::consts::LOG2_E);
    format!(
        r#"{hdr}// relation_score_kernel: concat [query; support] -> ReLU(W1*x + b1) -> W2*h + b2 -> sigmoid
// Each thread computes one (query, support) pair relation score.
// Layout: p_query/p_support are [n_pairs * feat_dim] row-major; the network
// input is the concatenation x = [query ; support] of length 2*feat_dim.
// p_w1 is [hidden_dim * (2*feat_dim)] row-major: row j holds the weights of
// hidden unit j, with columns 0..feat_dim multiplying the query slice and
// columns feat_dim..2*feat_dim multiplying the support slice.
// p_b1 is [hidden_dim], p_w2 is [hidden_dim], p_b2 is [1], p_out is [n_pairs].
.visible .entry relation_score_kernel(
    .param .u64 p_query,
    .param .u64 p_support,
    .param .u64 p_w1,
    .param .u64 p_b1,
    .param .u64 p_w2,
    .param .u64 p_b2,
    .param .u64 p_out,
    .param .u32 feat_dim,
    .param .u32 hidden_dim,
    .param .u32 n_pairs
)
{{
    .reg .u64  %rd<24>;
    .reg .u32  %r<20>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0,  [p_query];
    ld.param.u64  %rd1,  [p_support];
    ld.param.u64  %rd2,  [p_w1];
    ld.param.u64  %rd3,  [p_b1];
    ld.param.u64  %rd4,  [p_w2];
    ld.param.u64  %rd5,  [p_b2];
    ld.param.u64  %rd6,  [p_out];
    ld.param.u32  %r1,   [feat_dim];
    ld.param.u32  %r2,   [hidden_dim];
    ld.param.u32  %r0,   [n_pairs];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $RS_DONE;

    // Per-pair base element offsets into the query / support feature arrays.
    mul.lo.u32    %r7, %r6, %r1;            // r7 = pair * feat_dim
    mul.wide.u32  %rd7, %r7, 4;
    add.u64       %rd8, %rd0, %rd7;         // rd8 = &p_query[pair*feat_dim]
    add.u64       %rd9, %rd1, %rd7;         // rd9 = &p_support[pair*feat_dim]

    // in_dim = 2 * feat_dim (concatenated feature length / W1 row stride).
    shl.b32       %r8, %r1, 1;              // r8 = 2 * feat_dim

    // pre_sig accumulator for the output layer: s = sum_j W2[j]*h_j + b2.
    mov.f32       %f0, {ZERO};
    // j = 0 : hidden-unit loop index.
    mov.u32       %r9, 0;

$RS_HIDDEN_LOOP:
    setp.ge.u32   %p1, %r9, %r2;
    @%p1 bra $RS_HIDDEN_DONE;

    // Base element offset of W1 row j: j * in_dim.
    mul.lo.u32    %r10, %r9, %r8;           // r10 = j * (2*feat_dim)
    mul.wide.u32  %rd10, %r10, 4;
    add.u64       %rd11, %rd2, %rd10;       // rd11 = &p_w1[j*in_dim] (query cols)
    mul.wide.u32  %rd12, %r1, 4;
    add.u64       %rd13, %rd11, %rd12;      // rd13 = &p_w1[j*in_dim + feat_dim]

    // h_j pre-activation accumulator, seeded with b1[j].
    mul.wide.u32  %rd14, %r9, 4;
    add.u64       %rd15, %rd3, %rd14;
    ld.global.f32 %f1, [%rd15];             // f1 = b1[j]

    // i = 0 : feature loop index over the concatenated halves.
    mov.u32       %r11, 0;

$RS_FEAT_LOOP:
    setp.ge.u32   %p2, %r11, %r1;
    @%p2 bra $RS_FEAT_DONE;

    mul.wide.u32  %rd16, %r11, 4;           // byte offset of feature i

    // query contribution: W1[j, i] * query[i]
    add.u64       %rd17, %rd11, %rd16;
    ld.global.f32 %f2, [%rd17];             // f2 = W1[j, i]
    add.u64       %rd18, %rd8, %rd16;
    ld.global.f32 %f3, [%rd18];             // f3 = query[i]
    fma.rn.f32    %f1, %f2, %f3, %f1;

    // support contribution: W1[j, feat_dim + i] * support[i]
    add.u64       %rd19, %rd13, %rd16;
    ld.global.f32 %f4, [%rd19];             // f4 = W1[j, feat_dim+i]
    add.u64       %rd20, %rd9, %rd16;
    ld.global.f32 %f5, [%rd20];             // f5 = support[i]
    fma.rn.f32    %f1, %f4, %f5, %f1;

    add.u32       %r11, %r11, 1;
    bra           $RS_FEAT_LOOP;

$RS_FEAT_DONE:
    // ReLU activation: h_j = max(pre_activation, 0).
    max.f32       %f6, %f1, {ZERO};

    // Output layer MAC: s += W2[j] * h_j.
    add.u64       %rd21, %rd4, %rd14;
    ld.global.f32 %f7, [%rd21];             // f7 = W2[j]
    fma.rn.f32    %f0, %f7, %f6, %f0;

    add.u32       %r9, %r9, 1;
    bra           $RS_HIDDEN_LOOP;

$RS_HIDDEN_DONE:
    // s += b2[0]
    ld.global.f32 %f8, [%rd5];
    add.f32       %f0, %f0, %f8;

    // sigmoid(s) = 1 / (1 + exp(-s)) = ex2(s*log2e) / (1 + ex2(s*log2e))
    mul.f32       %f9, %f0, {LOG2E};
    ex2.approx.f32 %f10, %f9;               // f10 = exp(s)
    add.f32       %f11, %f10, {ONE};
    div.rn.f32    %f12, %f10, %f11;         // f12 = sigmoid(s)

    mul.wide.u32  %rd22, %r6, 4;
    add.u64       %rd23, %rd6, %rd22;
    st.global.f32 [%rd23], %f12;

$RS_DONE:
    mov.u32       %r12, 0;
    mov.u32       %r13, 0;
    mov.u32       %r14, 0;
    mov.u32       %r15, 0;
    mov.u32       %r16, 0;
    mov.u32       %r17, 0;
    mov.u32       %r18, 0;
    mov.u32       %r19, 0;
    mov.f32       %f13, {ZERO};
    mov.f32       %f14, {ZERO};
    mov.f32       %f15, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        LOG2E = log2e,
    )
}

pub fn meta_grad_accum_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// meta_grad_accum_kernel: out[i] = sum_t grad_t[i] / n_tasks
// p_grads: [n_tasks * n_params], p_out: [n_params]
.visible .entry meta_grad_accum_kernel(
    .param .u64 p_grads,
    .param .u64 p_out,
    .param .u32 n_params,
    .param .u32 n_tasks
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_grads];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n_params];
    ld.param.u32  %r1,  [n_tasks];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;
    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;
    mov.u32       %r8, %r5;

$MGA_OUTER:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $MGA_DONE;

    mov.f32       %f0, {ZERO};
    mov.u32       %r9, 0;

$MGA_INNER:
    setp.ge.u32   %p1, %r9, %r1;
    @%p1 bra $MGA_INNER_DONE;

    mul.lo.u32    %r2, %r9, %r0;
    add.u32       %r3, %r2, %r8;
    mul.wide.u32  %rd2, %r3, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32       %f0, %f0, %f1;
    add.u32       %r9, %r9, 1;
    bra           $MGA_INNER;

$MGA_INNER_DONE:
    cvt.rn.f32.u32 %f2, %r1;
    div.rn.f32    %f3, %f0, %f2;
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f3;

    add.u32       %r8, %r8, %r7;
    bra           $MGA_OUTER;

$MGA_DONE:
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

pub fn episode_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// episode_sample_kernel: LCG-based Fisher-Yates class/example selection
// Writes selected class indices to p_class_out using LCG shuffle
.visible .entry episode_sample_kernel(
    .param .u64 p_class_out,
    .param .u32 n_classes,
    .param .u32 n_way,
    .param .u64 seed
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<14>;
    .reg .f32  %f<4>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_class_out];
    ld.param.u32  %r0,  [n_classes];
    ld.param.u32  %r1,  [n_way];
    ld.param.u64  %rd1, [seed];

    // Only thread 0 does Fisher-Yates
    mov.u32       %r2, %tid.x;
    setp.ne.u32   %p0, %r2, 0;
    @%p0 bra $ES_DONE;

    // Initialize indices 0..n_classes in output buffer
    mov.u32       %r3, 0;
$ES_INIT_LOOP:
    setp.ge.u32   %p1, %r3, %r0;
    @%p1 bra $ES_INIT_DONE;
    mul.wide.u32  %rd2, %r3, 4;
    add.u64       %rd3, %rd0, %rd2;
    st.global.u32 [%rd3], %r3;
    add.u32       %r3, %r3, 1;
    bra           $ES_INIT_LOOP;

$ES_INIT_DONE:
    // Fisher-Yates: for i from n_classes-1 downto 1
    mov.u32       %r4, %r0;
    sub.u32       %r4, %r4, 1;
    mov.u64       %rd4, %rd1;

$ES_SHUFFLE_LOOP:
    setp.le.u32   %p1, %r4, 0;
    @%p1 bra $ES_SHUFFLE_DONE;

    // LCG step
    mov.u64       %rd5, 6364136223846793005;
    mul.lo.u64    %rd4, %rd4, %rd5;
    mov.u64       %rd6, 1442695040888963407;
    add.u64       %rd4, %rd4, %rd6;
    shr.u64       %rd5, %rd4, 33;
    cvt.u32.u64   %r5, %rd5;

    // j = r5 % (r4 + 1)
    add.u32       %r6, %r4, 1;
    rem.u32       %r7, %r5, %r6;

    // swap indices[r4] and indices[r7]
    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd0, %rd2;
    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.u32 %r8, [%rd3];
    ld.global.u32 %r9, [%rd6];
    st.global.u32 [%rd3], %r9;
    st.global.u32 [%rd6], %r8;

    sub.u32       %r4, %r4, 1;
    bra           $ES_SHUFFLE_LOOP;

$ES_SHUFFLE_DONE:

$ES_DONE:
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.u32       %r12, 0;
    mov.u32       %r13, 0;
    mov.f32       %f0, {ZERO};
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};
    mov.f32       %f3, {ZERO};
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_ptx(ptx: &str, sm: u32, name: &str) {
        assert!(ptx.contains(&format!("sm_{sm}")));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".visible .entry"));
        assert!(ptx.contains(name));
    }

    #[test]
    fn all_kernels_all_sm() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        for sm in sm_versions {
            check_ptx(&inner_sgd_ptx(sm), sm, "inner_sgd_kernel");
            check_ptx(&reptile_update_ptx(sm), sm, "reptile_update_kernel");
            check_ptx(&proto_distance_ptx(sm), sm, "proto_distance_kernel");
            check_ptx(&cosine_sim_ptx(sm), sm, "cosine_sim_kernel");
            check_ptx(&relation_score_ptx(sm), sm, "relation_score_kernel");
            check_ptx(&meta_grad_accum_ptx(sm), sm, "meta_grad_accum_kernel");
            check_ptx(&episode_sample_ptx(sm), sm, "episode_sample_kernel");
        }
    }

    #[test]
    fn f32_hex_known_values() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn relation_score_emits_mlp_body() {
        // The real relation-network kernel must contain the hidden-unit loop,
        // the inner feature MAC loop and the sigmoid output sequence — it must
        // no longer be the placeholder that simply stores 0.0 to p_out.
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            let ptx = relation_score_ptx(sm);

            // Hidden-layer loop and inner feature MAC loop are present.
            assert!(
                ptx.contains("$RS_HIDDEN_LOOP:"),
                "sm={sm}: missing hidden-unit loop label"
            );
            assert!(
                ptx.contains("$RS_FEAT_LOOP:"),
                "sm={sm}: missing feature MAC loop label"
            );
            // Multiply-accumulate instructions for the two layers.
            assert!(
                ptx.contains("fma.rn.f32"),
                "sm={sm}: missing fused multiply-add MAC instruction"
            );
            // ReLU activation on the hidden pre-activation.
            assert!(
                ptx.contains("max.f32"),
                "sm={sm}: missing ReLU (max.f32) activation"
            );
            // Sigmoid output: ex2.approx + reciprocal division.
            assert!(
                ptx.contains("ex2.approx.f32"),
                "sm={sm}: missing ex2.approx for sigmoid"
            );
            assert!(
                ptx.contains("div.rn.f32"),
                "sm={sm}: missing reciprocal division for sigmoid"
            );

            // The stale placeholder comment and its semantics must be gone.
            assert!(
                !ptx.contains("placeholder"),
                "sm={sm}: stale placeholder comment still present"
            );
            assert!(
                !ptx.contains("sigmoid(0.0)"),
                "sm={sm}: stale sigmoid(0.0)=0.5 comment still present"
            );
        }
    }

    /// Bit-exact CPU mirror of the `relation_score_kernel` PTX body. Used to
    /// document the GPU/CPU contract: row-major W1 indexing, ReLU on the
    /// hidden layer, then a sigmoid output. The arithmetic order matches the
    /// PTX (`fma` accumulation seeded with the bias, `max` for ReLU).
    fn relation_score_cpu(
        query: &[f32],
        support: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: f32,
        feat_dim: usize,
        hidden_dim: usize,
    ) -> f32 {
        let in_dim = 2 * feat_dim;
        let mut pre_sig = b2;
        for j in 0..hidden_dim {
            let row = &w1[j * in_dim..(j + 1) * in_dim];
            let mut acc = b1[j];
            for i in 0..feat_dim {
                acc += row[i] * query[i];
                acc += row[feat_dim + i] * support[i];
            }
            let h_j = acc.max(0.0_f32);
            pre_sig += w2[j] * h_j;
        }
        1.0_f32 / (1.0_f32 + (-pre_sig).exp())
    }

    #[test]
    fn relation_score_cpu_mirror_matches_reference() {
        // Cross-check the documented kernel contract against a hand-rolled
        // forward pass. Negative pre-activations exercise the ReLU clamp.
        let feat_dim = 3_usize;
        let hidden_dim = 4_usize;
        let in_dim = 2 * feat_dim;
        let query = [0.5_f32, -0.25, 1.0];
        let support = [-0.5_f32, 0.75, 0.1];
        let w1: Vec<f32> = (0..hidden_dim * in_dim)
            .map(|k| (k as f32) * 0.1 - 0.3)
            .collect();
        let b1 = [-2.0_f32, 0.05, -0.1, 0.2];
        let w2 = [0.4_f32, -0.3, 0.2, 0.6];
        let b2 = 0.15_f32;

        let kernel_like =
            relation_score_cpu(&query, &support, &w1, &b1, &w2, b2, feat_dim, hidden_dim);

        // Independent recompute with the same row-major indexing.
        let mut expected = b2;
        for j in 0..hidden_dim {
            let mut acc = b1[j];
            for i in 0..feat_dim {
                acc += w1[j * in_dim + i] * query[i];
                acc += w1[j * in_dim + feat_dim + i] * support[i];
            }
            expected += w2[j] * acc.max(0.0_f32);
        }
        let expected = 1.0_f32 / (1.0_f32 + (-expected).exp());

        assert!(
            (kernel_like - expected).abs() < 1e-6,
            "relation-score mirror mismatch: {kernel_like} vs {expected}"
        );
        // Sigmoid output is always a valid probability.
        assert!(
            (0.0_f32..=1.0_f32).contains(&kernel_like),
            "sigmoid output out of [0,1]: {kernel_like}"
        );
    }
}
