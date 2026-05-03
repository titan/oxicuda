//! PTX GPU kernel sources for Neural Architecture Search operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`arch_softmax_ptx`] | Numerically-stable softmax over architecture parameters |
//! | [`mixed_op_blend_ptx`] | Weighted sum of K operator outputs |
//! | [`gumbel_softmax_ptx`] | Gumbel-softmax with temperature annealing |
//! | [`flops_accumulate_ptx`] | Per-layer FLOPs accumulation with arch-prob weights |
//! | [`pareto_dominate_ptx`] | Pareto dominance relation for NSGA-II |
//! | [`arch_grad_ptx`] | Architecture parameter gradient via softmax Jacobian |
//! | [`crossover_uniform_ptx`] | Uniform crossover for evolutionary NAS |

// ─── PTX header helper ────────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Convert a `f32` literal to its PTX hex representation.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: arch_softmax ───────────────────────────────────────────────────

/// Numerically-stable softmax over `n` architecture parameters.
///
/// Computes `exp(x_i - max_x) / Σ exp(x_j - max_x)` using a grid-stride loop.
/// Each block processes one independent softmax (one edge's arch params).
#[must_use]
pub fn arch_softmax_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{hdr}.visible .entry arch_softmax_kernel(
    .param .u64 p_logits,
    .param .u64 p_output,
    .param .u32 n
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_output];
    ld.param.u32  %r0,  [n];

    // tid = blockDim.x * blockIdx.x + threadIdx.x  (grid-stride)
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;        // r4 = tid

    // Phase 1: find max over all elements (each thread scans all)
    // This kernel assumes n <= 1024 and a single block handles one softmax
    mov.f32       %f0,  {neg_inf};             // max_val = -inf
    mov.u32       %r5,  0;                     // i = 0

$ASOFT_MAX_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $ASOFT_MAX_END;
    mul.wide.u32  %rd2, %r5, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1,  [%rd3];
    max.f32       %f0,  %f0, %f1;
    add.u32       %r5,  %r5, 1;
    bra           $ASOFT_MAX_LOOP;

$ASOFT_MAX_END:
    // Phase 2: compute sum of exp(x_i - max)
    mov.f32       %f2,  {zero};               // sum = 0
    mov.u32       %r6,  0;                     // i = 0

$ASOFT_SUM_LOOP:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $ASOFT_SUM_END;
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f3,  [%rd5];
    sub.f32       %f4,  %f3, %f0;             // x_i - max
    // exp via lg2: exp(x) = 2^(x * log2(e))
    mul.f32       %f5,  %f4, 0F3FB8AA3B;      // * log2(e)
    ex2.approx.f32 %f6, %f5;
    add.f32       %f2,  %f2, %f6;
    add.u32       %r6,  %r6, 1;
    bra           $ASOFT_SUM_LOOP;

$ASOFT_SUM_END:
    // Phase 3: each thread writes its own element (tid-indexed)
    setp.ge.u32   %p1, %r4, %r0;
    @%p1 bra $ASOFT_DONE;

    mul.wide.u32  %rd6, %r4, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f7,  [%rd7];
    sub.f32       %f8,  %f7, %f0;
    mul.f32       %f9,  %f8, 0F3FB8AA3B;
    ex2.approx.f32 %f10, %f9;
    div.rn.f32    %f11, %f10, %f2;
    add.u64       %rd8, %rd1, %rd6;
    st.global.f32 [%rd8], %f11;

$ASOFT_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: mixed_op_blend ─────────────────────────────────────────────────

/// Weighted sum of K operator outputs: `out[i] = Σ_k w[k] * ops_out[k*n + i]`.
#[must_use]
pub fn mixed_op_blend_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry mixed_op_blend_kernel(
    .param .u64 p_weights,
    .param .u64 p_ops_out,
    .param .u64 p_output,
    .param .u32 n_elems,
    .param .u32 n_ops
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_weights];
    ld.param.u64  %rd1, [p_ops_out];
    ld.param.u64  %rd2, [p_output];
    ld.param.u32  %r0,  [n_elems];
    ld.param.u32  %r1,  [n_ops];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;        // r5 = tid (element index)

    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $BLEND_DONE;

    // acc = 0
    mov.f32       %f0,  {zero};

    // Loop over ops: k in [0, n_ops)
    mov.u32       %r6,  0;                    // k = 0

$BLEND_OP_LOOP:
    setp.ge.u32   %p1, %r6, %r1;
    @%p1 bra $BLEND_OP_END;

    // w[k]
    mul.wide.u32  %rd3, %r6, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1,  [%rd4];              // f1 = w[k]

    // ops_out[k * n_elems + tid]
    mul.lo.u32    %r7,  %r6, %r0;            // k * n_elems
    add.u32       %r8,  %r7, %r5;            // + tid
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f2,  [%rd6];              // f2 = op_out

    fma.rn.f32    %f0,  %f1, %f2, %f0;      // acc += w[k] * op_out

    add.u32       %r6,  %r6, 1;
    bra           $BLEND_OP_LOOP;

$BLEND_OP_END:
    mul.wide.u32  %rd7, %r5, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

$BLEND_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: gumbel_softmax ─────────────────────────────────────────────────

/// Gumbel-softmax: `y_i = exp((log(π_i) + g_i)/τ) / Σ exp(...)`.
///
/// Gumbel noise via `-log(-log(u + ε))` transform, `u ~ Uniform(0,1)`.
/// Uses `lg2.approx.f32` + `ex2.approx.f32` for fast exp/log in PTX.
#[must_use]
pub fn gumbel_softmax_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let eps = f32_hex(1e-10_f32);
    let zero = f32_hex(0.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    // log2(e) for converting natural log to log base 2
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    // 1 / ln(2) = log2(e) — same constant, used for lg2 → ln conversion
    let inv_ln2 = f32_hex(1.0_f32 / std::f32::consts::LN_2);
    format!(
        r#"{hdr}.visible .entry gumbel_softmax_kernel(
    .param .u64 p_logits,
    .param .u64 p_uniform,
    .param .u64 p_output,
    .param .u32 n,
    .param .f32 temperature
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<20>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_uniform];
    ld.param.u64  %rd2, [p_output];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [temperature];

    // tid
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;

    // Phase 1 (serial): compute perturbed logits and find max
    mov.f32       %f1,  {neg_inf};            // max_val = -inf
    mov.u32       %r5,  0;

$GUMB_PREP_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $GUMB_PREP_END;
    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f2,  [%rd4];              // logit[i]
    // Gumbel noise: g = -ln(-ln(u + eps) + eps)
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f3,  [%rd5];              // u (uniform)
    add.f32       %f4,  %f3, {eps};          // u + eps
    // lg2(u+eps) then multiply by ln(2) to get ln(u+eps)
    lg2.approx.f32 %f5, %f4;                 // log2(u+eps)
    mul.f32       %f6,  %f5, {inv_ln2};      // * 1/log2(e) = ln(u+eps)
    // negate: -ln(u+eps)
    neg.f32       %f7,  %f6;
    add.f32       %f8,  %f7, {eps};          // + eps
    lg2.approx.f32 %f9, %f8;
    mul.f32       %f10, %f9, {inv_ln2};      // ln(-ln(u+eps)+eps)
    neg.f32       %f11, %f10;                // g = -ln(-ln(u+eps)+eps)
    // perturbed = (logit + gumbel) / temperature
    add.f32       %f12, %f2, %f11;
    div.rn.f32    %f13, %f12, %f0;
    max.f32       %f1,  %f1, %f13;
    add.u32       %r5,  %r5, 1;
    bra           $GUMB_PREP_LOOP;

$GUMB_PREP_END:
    // Phase 2: sum of exp(perturbed - max)
    mov.f32       %f14, {zero};
    mov.u32       %r6,  0;

$GUMB_SUM_LOOP:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $GUMB_SUM_END;
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f2,  [%rd7];
    add.u64       %rd8, %rd1, %rd6;
    ld.global.f32 %f3,  [%rd8];
    add.f32       %f4,  %f3, {eps};
    lg2.approx.f32 %f5, %f4;
    mul.f32       %f6,  %f5, {inv_ln2};
    neg.f32       %f7,  %f6;
    add.f32       %f8,  %f7, {eps};
    lg2.approx.f32 %f9, %f8;
    mul.f32       %f10, %f9, {inv_ln2};
    neg.f32       %f11, %f10;
    add.f32       %f12, %f2, %f11;
    div.rn.f32    %f13, %f12, %f0;
    sub.f32       %f15, %f13, %f1;
    mul.f32       %f16, %f15, {log2e};
    ex2.approx.f32 %f17, %f16;
    add.f32       %f14, %f14, %f17;
    add.u32       %r6,  %r6, 1;
    bra           $GUMB_SUM_LOOP;

$GUMB_SUM_END:
    // Phase 3: each thread writes its element
    setp.ge.u32   %p1, %r4, %r0;
    @%p1 bra $GUMB_DONE;

    mul.wide.u32  %rd9, %r4, 4;
    add.u64       %rd10, %rd0, %rd9;
    ld.global.f32 %f2,  [%rd10];
    add.u64       %rd11, %rd1, %rd9;
    ld.global.f32 %f3,  [%rd11];
    add.f32       %f4,  %f3, {eps};
    lg2.approx.f32 %f5, %f4;
    mul.f32       %f6,  %f5, {inv_ln2};
    neg.f32       %f7,  %f6;
    add.f32       %f8,  %f7, {eps};
    lg2.approx.f32 %f9, %f8;
    mul.f32       %f10, %f9, {inv_ln2};
    neg.f32       %f11, %f10;
    add.f32       %f12, %f2, %f11;
    div.rn.f32    %f13, %f12, %f0;
    sub.f32       %f15, %f13, %f1;
    mul.f32       %f16, %f15, {log2e};
    ex2.approx.f32 %f17, %f16;
    div.rn.f32    %f18, %f17, %f14;
    add.u64       %rd12, %rd2, %rd9;
    st.global.f32 [%rd12], %f18;

$GUMB_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 4: flops_accumulate ───────────────────────────────────────────────

/// Per-layer FLOPs accumulation: `total_flops += ops[i] * weight[i]`.
///
/// Uses `atom.global.add.f32` for concurrent accumulation across threads.
#[must_use]
pub fn flops_accumulate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry flops_accumulate_kernel(
    .param .u64 p_flops,
    .param .u64 p_weights,
    .param .u64 p_total,
    .param .u32 n
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<12>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_flops];
    ld.param.u64  %rd1, [p_weights];
    ld.param.u64  %rd2, [p_total];
    ld.param.u32  %r0,  [n];

    // tid = blockDim.x * blockIdx.x + threadIdx.x  (grid-stride)
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;

$FLOPS_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $FLOPS_DONE;

    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0,  [%rd4];              // flops[tid]

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1,  [%rd5];              // weight[tid]

    mul.rn.f32    %f2,  %f0, %f1;            // flops * weight
    atom.global.add.f32 %f3, [%rd2], %f2;   // atomic add to total

    // grid-stride advance
    mov.u32       %r5,  %ntid.x;
    mov.u32       %r6,  %nctaid.x;
    mul.lo.u32    %r7,  %r5, %r6;
    add.u32       %r4,  %r4, %r7;
    bra           $FLOPS_LOOP;

$FLOPS_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: pareto_dominate ────────────────────────────────────────────────

/// Pareto dominance matrix: `dom[i*n+j] = 1` if solution `i` dominates `j`.
///
/// Solution `i` dominates `j` iff all objectives of `i` ≤ objectives of `j`
/// and at least one objective strictly `<`.  Objectives layout: `obj[i*m+k]`.
#[must_use]
pub fn pareto_dominate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry pareto_dominate_kernel(
    .param .u64 p_objectives,
    .param .u64 p_domination,
    .param .u32 n_solutions,
    .param .u32 n_objectives
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<24>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1, %p2, %p3;

    ld.param.u64  %rd0, [p_objectives];
    ld.param.u64  %rd1, [p_domination];
    ld.param.u32  %r0,  [n_solutions];
    ld.param.u32  %r1,  [n_objectives];

    // Each thread handles one (i, j) pair
    // tid = i * n_solutions + j  (row-major)
    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;       // r5 = linear tid

    // total pairs = n^2
    mul.lo.u32    %r6,  %r0, %r0;
    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $PARETO_DONE;

    // i = tid / n,  j = tid % n
    div.u32       %r7,  %r5, %r0;            // i
    rem.u32       %r8,  %r5, %r0;            // j

    // skip diagonal (i == j)
    setp.eq.u32   %p0, %r7, %r8;
    @%p0 bra $PARETO_WRITE_ZERO;

    // Check dominance of i over j:
    // all_leq = 1 (all obj_i <= obj_j), any_lt = 0 (no strict <)
    // base_i = i * n_objectives, base_j = j * n_objectives
    mul.lo.u32    %r9,  %r7, %r1;            // base_i
    mul.lo.u32    %r10, %r8, %r1;            // base_j

    mov.u32       %r11, 1;                   // all_leq = true (1)
    mov.u32       %r12, 0;                   // any_lt  = false (0)
    mov.u32       %r13, 0;                   // k = 0

$PARETO_OBJ_LOOP:
    setp.ge.u32   %p1, %r13, %r1;
    @%p1 bra $PARETO_OBJ_END;

    // obj_i[k]
    add.u32       %r14, %r9,  %r13;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0,  [%rd4];

    // obj_j[k]
    add.u32       %r15, %r10, %r13;
    mul.wide.u32  %rd5, %r15, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f1,  [%rd6];

    // if obj_i[k] > obj_j[k]: all_leq = 0
    setp.gt.f32   %p2, %f0, %f1;
    @%p2 mov.u32  %r11, 0;

    // if obj_i[k] < obj_j[k]: any_lt = 1
    setp.lt.f32   %p3, %f0, %f1;
    @%p3 mov.u32  %r12, 1;

    add.u32       %r13, %r13, 1;
    bra           $PARETO_OBJ_LOOP;

$PARETO_OBJ_END:
    // dominates = all_leq AND any_lt
    and.b32       %r16, %r11, %r12;
    mul.wide.u32  %rd7, %r5, 4;
    add.u64       %rd8, %rd1, %rd7;
    st.global.u32 [%rd8], %r16;
    bra           $PARETO_DONE;

$PARETO_WRITE_ZERO:
    mul.wide.u32  %rd9, %r5, 4;
    add.u64       %rd10, %rd1, %rd9;
    mov.u32       %r17, 0;
    st.global.u32 [%rd10], %r17;

$PARETO_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 6: arch_grad ──────────────────────────────────────────────────────

/// Architecture parameter gradient via softmax Jacobian diagonal.
///
/// `grad_alpha[k] = w[k] * (1 - w[k]) * Σ_i (out_grad[i] * op_k_out[i])`.
#[must_use]
pub fn arch_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}.visible .entry arch_grad_kernel(
    .param .u64 p_weights,
    .param .u64 p_out_grad,
    .param .u64 p_op_outputs,
    .param .u64 p_grad_alpha,
    .param .u32 n_ops,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_weights];
    ld.param.u64  %rd1, [p_out_grad];
    ld.param.u64  %rd2, [p_op_outputs];
    ld.param.u64  %rd3, [p_grad_alpha];
    ld.param.u32  %r0,  [n_ops];
    ld.param.u32  %r1,  [n_elems];

    // Each thread handles one op index k
    mov.u32       %r2,  %ntid.x;
    mov.u32       %r3,  %ctaid.x;
    mov.u32       %r4,  %tid.x;
    mad.lo.u32    %r5,  %r2, %r3, %r4;       // k = tid

    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $AGRAD_DONE;

    // w[k]
    mul.wide.u32  %rd4, %r5, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0,  [%rd5];              // w_k

    // dot = Σ_i (out_grad[i] * op_k_out[i])
    mov.f32       %f1,  {zero};              // dot = 0
    mov.u32       %r6,  0;                   // i = 0

$AGRAD_DOT_LOOP:
    setp.ge.u32   %p1, %r6, %r1;
    @%p1 bra $AGRAD_DOT_END;

    // out_grad[i]
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f2,  [%rd7];

    // op_k_out[k * n_elems + i]
    mul.lo.u32    %r7,  %r5, %r1;
    add.u32       %r8,  %r7, %r6;
    mul.wide.u32  %rd8, %r8, 4;
    add.u64       %rd9, %rd2, %rd8;
    ld.global.f32 %f3,  [%rd9];

    fma.rn.f32    %f1,  %f2, %f3, %f1;      // dot += grad * op_out

    add.u32       %r6,  %r6, 1;
    bra           $AGRAD_DOT_LOOP;

$AGRAD_DOT_END:
    // grad_alpha[k] = w_k * (1 - w_k) * dot
    sub.f32       %f4,  {one}, %f0;          // 1 - w_k
    mul.rn.f32    %f5,  %f0, %f4;            // w_k * (1 - w_k)
    mul.rn.f32    %f6,  %f5, %f1;            // * dot
    add.u64       %rd10, %rd3, %rd4;
    st.global.f32 [%rd10], %f6;

$AGRAD_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 7: crossover_uniform ─────────────────────────────────────────────

/// Uniform crossover: `child[i] = (mask[i] > 0) ? parent_a[i] : parent_b[i]`.
///
/// `mask` is a 0/1 integer array; `parent_a/b` and `child` are `u32` arrays
/// (gene = op index per edge).
#[must_use]
pub fn crossover_uniform_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry crossover_uniform_kernel(
    .param .u64 p_parent_a,
    .param .u64 p_parent_b,
    .param .u64 p_mask,
    .param .u64 p_child,
    .param .u32 n_genes
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_parent_a];
    ld.param.u64  %rd1, [p_parent_b];
    ld.param.u64  %rd2, [p_mask];
    ld.param.u64  %rd3, [p_child];
    ld.param.u32  %r0,  [n_genes];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1,  %ntid.x;
    mov.u32       %r2,  %ctaid.x;
    mov.u32       %r3,  %tid.x;
    mad.lo.u32    %r4,  %r1, %r2, %r3;

$CROSS_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $CROSS_DONE;

    mul.wide.u32  %rd4, %r4, 4;

    // mask[tid]
    add.u64       %rd5, %rd2, %rd4;
    ld.global.u32 %r5,  [%rd5];

    // parent_a[tid]
    add.u64       %rd6, %rd0, %rd4;
    ld.global.u32 %r6,  [%rd6];

    // parent_b[tid]
    add.u64       %rd7, %rd1, %rd4;
    ld.global.u32 %r7,  [%rd7];

    // child[tid] = mask ? parent_a : parent_b
    setp.ne.u32   %p1, %r5, 0;
    selp.u32      %r8,  %r6, %r7, %p1;

    add.u64       %rd8, %rd3, %rd4;
    st.global.u32 [%rd8], %r8;

    // grid-stride advance
    mov.u32       %r9,  %ntid.x;
    mov.u32       %r10, %nctaid.x;
    mul.lo.u32    %r11, %r9, %r10;
    add.u32       %r4,  %r4, %r11;
    bra           $CROSS_LOOP;

$CROSS_DONE:
    ret;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_ptx(ptx: &str, sm: u32) {
        assert!(ptx.contains(".version"), "missing .version for sm_{sm}");
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm}"
        );
        assert!(
            ptx.contains(".address_size 64"),
            "missing .address_size 64 for sm_{sm}"
        );
    }

    #[test]
    fn all_kernels_all_sm_versions() {
        for &sm in SM_VERSIONS {
            check_ptx(&arch_softmax_ptx(sm), sm);
            check_ptx(&mixed_op_blend_ptx(sm), sm);
            check_ptx(&gumbel_softmax_ptx(sm), sm);
            check_ptx(&flops_accumulate_ptx(sm), sm);
            check_ptx(&pareto_dominate_ptx(sm), sm);
            check_ptx(&arch_grad_ptx(sm), sm);
            check_ptx(&crossover_uniform_ptx(sm), sm);
        }
    }

    #[test]
    fn arch_softmax_contains_ex2() {
        for &sm in SM_VERSIONS {
            let ptx = arch_softmax_ptx(sm);
            assert!(ptx.contains("ex2.approx.f32"), "missing ex2 for sm_{sm}");
        }
    }

    #[test]
    fn gumbel_softmax_contains_lg2_and_ex2() {
        for &sm in SM_VERSIONS {
            let ptx = gumbel_softmax_ptx(sm);
            assert!(ptx.contains("lg2.approx.f32"), "missing lg2 for sm_{sm}");
            assert!(ptx.contains("ex2.approx.f32"), "missing ex2 for sm_{sm}");
        }
    }

    #[test]
    fn flops_accumulate_contains_atom_add() {
        for &sm in SM_VERSIONS {
            let ptx = flops_accumulate_ptx(sm);
            assert!(
                ptx.contains("atom.global.add.f32"),
                "missing atom.add for sm_{sm}"
            );
        }
    }

    #[test]
    fn crossover_uses_selp() {
        for &sm in SM_VERSIONS {
            let ptx = crossover_uniform_ptx(sm);
            assert!(ptx.contains("selp.u32"), "missing selp.u32 for sm_{sm}");
        }
    }
}
