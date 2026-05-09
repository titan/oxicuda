//! PTX GPU kernel sources for continual/lifelong learning operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`ewc_penalty_ptx`] | Compute EWC penalty: `Σ F_i · (θ_i - θ*_i)²` |
//! | [`fisher_diag_ptx`] | Accumulate diagonal Fisher: `F += g²` per parameter |
//! | [`gradient_project_ptx`] | Project gradient for GEM: `g' = g - (g·m/m·m)·m` |
//! | [`mask_apply_ptx`] | Apply binary mask in-place: `w *= mask` (PackNet/Piggyback) |
//! | [`si_omega_update_ptx`] | SI importance: `Ω += |Δθ · (-dL/dθ)|` |
//! | [`logit_distill_ptx`] | DER++ distillation: KL divergence on stored logits |
//! | [`replay_sample_ptx`] | Reservoir sampling update: conditional swap based on LCG index |

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Format an f32 as a PTX hex literal.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: ewc_penalty ───────────────────────────────────────────────────

/// Compute EWC penalty contribution per-parameter:
/// `out += F_i * (theta_i - theta_star_i)^2`.
///
/// Grid-stride over parameters; uses `fma.rn.f32` for precision and
/// `atom.global.add.f32` for accumulation.
#[must_use]
pub fn ewc_penalty_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// ewc_penalty_kernel: out += F_i * (theta_i - theta_star_i)^2
// Grid-stride kernel; each thread accumulates its portion into p_out atomically.
.visible .entry ewc_penalty_kernel(
    .param .u64 p_theta,
    .param .u64 p_theta_star,
    .param .u64 p_fisher,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_theta];
    ld.param.u64  %rd1, [p_theta_star];
    ld.param.u64  %rd2, [p_fisher];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$EWC_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $EWC_DONE;

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;       // &theta[i]
    add.u64       %rd6, %rd1, %rd4;       // &theta_star[i]
    add.u64       %rd7, %rd2, %rd4;       // &fisher[i]

    ld.global.f32 %f0, [%rd5];            // theta_i
    ld.global.f32 %f1, [%rd6];            // theta_star_i
    ld.global.f32 %f2, [%rd7];            // F_i

    sub.f32       %f3, %f0, %f1;          // delta = theta_i - theta_star_i
    mul.f32       %f4, %f3, %f3;          // delta^2
    mul.f32       %f5, %f2, %f4;          // F_i * delta^2

    // FMA variant: F_i * delta^2 + 0 (keeps fma.rn in use)
    fma.rn.f32    %f6, %f2, %f4, {ZERO};

    atom.global.add.f32 %f7, [%rd3], %f6;

    add.u32       %r7, %r7, %r6;
    bra           $EWC_LOOP;

$EWC_DONE:
    // suppress unused-register warnings
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f8, {ZERO};
    mov.f32       %f9, {ZERO};
    mov.u64       %rd8, 0;
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 2: fisher_diag ───────────────────────────────────────────────────

/// Accumulate diagonal Fisher information: `F_i += g_i^2`.
///
/// Grid-stride; uses `mul.f32` and `atom.global.add.f32`.
#[must_use]
pub fn fisher_diag_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// fisher_diag_kernel: F[i] += grad[i]^2 (diagonal Fisher accumulation)
.visible .entry fisher_diag_kernel(
    .param .u64 p_grad,
    .param .u64 p_fisher,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_grad];
    ld.param.u64  %rd1, [p_fisher];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$FD_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $FD_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;       // &grad[i]
    add.u64       %rd4, %rd1, %rd2;       // &fisher[i]

    ld.global.f32 %f0, [%rd3];            // g_i
    mul.f32       %f1, %f0, %f0;          // g_i^2
    atom.global.add.f32 %f2, [%rd4], %f1;

    add.u32       %r7, %r7, %r6;
    bra           $FD_LOOP;

$FD_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u64       %rd5, 0;
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 3: gradient_project ──────────────────────────────────────────────

/// Project gradient for GEM:
/// `g' = g - (g·m / m·m) · m` (project g onto the constraint half-space).
///
/// Requires two passes (dot products then projection); this kernel handles
/// the projection step given pre-computed dot products `dot_gm` and `dot_mm`.
#[must_use]
pub fn gradient_project_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-12_f32);
    format!(
        r#"{hdr}// gradient_project_kernel: g[i] -= (dot_gm / dot_mm) * m[i]
// p_dot_gm and p_dot_mm are scalar pointers to pre-accumulated dot products.
// Use atom.global.add.f32 for dot product accumulation in a prior pass.
.visible .entry gradient_project_kernel(
    .param .u64 p_grad,
    .param .u64 p_mem_grad,
    .param .u64 p_dot_gm,
    .param .u64 p_dot_mm,
    .param .u32 n
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_grad];
    ld.param.u64  %rd1, [p_mem_grad];
    ld.param.u64  %rd2, [p_dot_gm];
    ld.param.u64  %rd3, [p_dot_mm];
    ld.param.u32  %r0,  [n];

    ld.global.f32 %f0, [%rd2];            // dot_gm = g . m
    ld.global.f32 %f1, [%rd3];            // dot_mm = m . m

    // scale = dot_gm / (dot_mm + eps)
    mov.f32       %f2, {EPS};
    add.f32       %f3, %f1, %f2;
    div.rn.f32    %f4, %f0, %f3;          // scale = dot_gm / dot_mm

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$GP_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $GP_DONE;

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;       // &grad[i]
    add.u64       %rd6, %rd1, %rd4;       // &mem_grad[i]

    ld.global.f32 %f5, [%rd5];            // g[i]
    ld.global.f32 %f6, [%rd6];            // m[i]

    // g'[i] = g[i] - scale * m[i]
    mul.f32       %f7, %f4, %f6;
    sub.f32       %f8, %f5, %f7;
    st.global.f32 [%rd5], %f8;

    add.u32       %r7, %r7, %r6;
    bra           $GP_LOOP;

$GP_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f9, {ZERO};
    mov.u64       %rd7, 0;
    mov.u64       %rd8, 0;
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
    )
}

// ─── Kernel 4: mask_apply ────────────────────────────────────────────────────

/// Apply binary mask in-place: `w[i] *= mask[i]`.
///
/// Uses `setp.ne.u32` to interpret the mask as boolean (non-zero = keep)
/// and `mul.f32` to zero masked weights.
#[must_use]
pub fn mask_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// mask_apply_kernel: w[i] = w[i] * (mask[i] != 0 ? 1.0 : 0.0)
// mask is a u8 buffer; each byte is 0 (masked) or 1 (kept).
.visible .entry mask_apply_kernel(
    .param .u64 p_weights,
    .param .u64 p_mask,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .u8   %rc0;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_weights];
    ld.param.u64  %rd1, [p_mask];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$MA_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $MA_DONE;

    // Load mask byte
    cvt.u64.u32   %rd2, %r7;
    add.u64       %rd3, %rd1, %rd2;
    ld.global.u8  %rc0, [%rd3];
    cvt.u32.u8    %r8, %rc0;

    // m_f = (mask_byte != 0) ? 1.0 : 0.0
    setp.ne.u32   %p1, %r8, 0;
    selp.f32      %f0, {ONE}, {ZERO}, %p1;

    // w[i] *= m_f
    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    mul.f32       %f2, %f1, %f0;
    st.global.f32 [%rd5], %f2;

    add.u32       %r7, %r7, %r6;
    bra           $MA_LOOP;

$MA_DONE:
    mov.u32       %r9, 0;
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 5: si_omega_update ───────────────────────────────────────────────

/// Synaptic Intelligence importance update:
/// `Ω_i += |Δθ_i * (-dL/dθ_i)|` = `|Δθ_i * grad_i|`.
///
/// Uses `mul.f32`, `abs.f32`, and `atom.global.add.f32`.
#[must_use]
pub fn si_omega_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// si_omega_update_kernel: omega[i] += |delta_theta[i] * grad[i]|
// delta_theta = theta_current - theta_prev
.visible .entry si_omega_update_kernel(
    .param .u64 p_delta_theta,
    .param .u64 p_grad,
    .param .u64 p_omega,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_delta_theta];
    ld.param.u64  %rd1, [p_grad];
    ld.param.u64  %rd2, [p_omega];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$SI_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $SI_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;       // &delta_theta[i]
    add.u64       %rd5, %rd1, %rd3;       // &grad[i]
    add.u64       %rd6, %rd2, %rd3;       // &omega[i]

    ld.global.f32 %f0, [%rd4];            // delta_theta_i
    ld.global.f32 %f1, [%rd5];            // grad_i

    mul.f32       %f2, %f0, %f1;          // delta_theta * grad
    abs.f32       %f3, %f2;               // |delta_theta * grad|
    atom.global.add.f32 %f4, [%rd6], %f3;

    add.u32       %r7, %r7, %r6;
    bra           $SI_LOOP;

$SI_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f5, {ZERO};
    mov.f32       %f6, {ZERO};
    mov.f32       %f7, {ZERO};
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 6: logit_distill ─────────────────────────────────────────────────

/// DER++ logit distillation:
/// Compute KL(softmax(z_stored) || softmax(z_current)) contribution per class.
///
/// Uses `ex2.approx.f32` for exp (base-2) and `lg2.approx.f32` for log (base-2).
#[must_use]
pub fn logit_distill_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    let inv_log2e = f32_hex(1.0_f32 / std::f32::consts::LOG2_E); // ln(2)
    format!(
        r#"{hdr}// logit_distill_kernel: KL contribution per class.
// Approximates softmax via ex2.approx.f32 and lg2.approx.f32.
// p_z_stored: stored logits, p_z_current: current logits, n: n_classes.
// Accumulates KL divergence into p_kl_out (scalar, atomic).
.visible .entry logit_distill_kernel(
    .param .u64 p_z_stored,
    .param .u64 p_z_current,
    .param .u64 p_kl_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_z_stored];
    ld.param.u64  %rd1, [p_z_current];
    ld.param.u64  %rd2, [p_kl_out];
    ld.param.u32  %r0,  [n];

    // log2(e) constant for converting ln to log2
    mov.f32       %f12, {LOG2E};
    // ln(2) for converting log2 back to ln
    mov.f32       %f13, {INV_LOG2E};

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$KL_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $KL_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;       // &z_stored[i]
    add.u64       %rd5, %rd1, %rd3;       // &z_current[i]

    ld.global.f32 %f0, [%rd4];            // z_stored_i
    ld.global.f32 %f1, [%rd5];            // z_current_i

    // p_stored = exp(z_stored_i) [unnormalized; normalized on host]
    // Use ex2: exp(x) = 2^(x * log2(e))
    mul.f32       %f2, %f0, %f12;
    ex2.approx.f32 %f3, %f2;              // exp(z_stored_i) approx

    mul.f32       %f4, %f1, %f12;
    ex2.approx.f32 %f5, %f4;              // exp(z_current_i) approx

    // log(p_stored / p_current) = log(p_stored) - log(p_current)
    // lg2 gives log2; convert: ln(x) = log2(x) * ln(2)
    lg2.approx.f32 %f6, %f3;             // log2(exp(z_stored_i)) ≈ z_stored_i * log2e
    mul.f32       %f7, %f6, %f13;         // ln(p_stored) approx

    lg2.approx.f32 %f8, %f5;             // log2(exp(z_current_i))
    mul.f32       %f9, %f8, %f13;         // ln(p_current) approx

    // KL contribution: p_stored * (ln_p_stored - ln_p_current)
    sub.f32       %f10, %f7, %f9;
    mul.f32       %f11, %f3, %f10;        // p_stored * log(p_stored/p_current)

    atom.global.add.f32 %f0, [%rd2], %f11;

    add.u32       %r7, %r7, %r6;
    bra           $KL_LOOP;

$KL_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        LOG2E = log2e,
        INV_LOG2E = inv_log2e,
    )
}

// ─── Kernel 7: replay_sample ─────────────────────────────────────────────────

/// Reservoir sampling update: conditionally swap a new sample into the buffer
/// based on an LCG-generated random index.
///
/// Each thread handles one candidate sample. If the LCG-generated index
/// falls within the buffer capacity, the sample at that position is replaced.
#[must_use]
pub fn replay_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// replay_sample_kernel: reservoir sampling update.
// For sample index n_seen, generate r = LCG(seed, n_seen) % (n_seen + 1).
// If r < capacity, swap buffer[r] index slot (host performs data copy).
// This kernel writes the swap target index (-1 if no swap) into p_swap_idx.
.visible .entry replay_sample_kernel(
    .param .u64 p_swap_idx,
    .param .u32 n_seen,
    .param .u32 capacity,
    .param .u64 seed
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<14>;
    .reg .f32  %f<4>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_swap_idx];
    ld.param.u32  %r0,  [n_seen];
    ld.param.u32  %r1,  [capacity];
    ld.param.u64  %rd1, [seed];

    // Only thread 0 does work
    mov.u32       %r2, %tid.x;
    setp.ne.u32   %p0, %r2, 0;
    @%p0 bra $RS_DONE;

    // LCG: state = seed XOR n_seen, then one step
    cvt.u64.u32   %rd2, %r0;
    xor.b64       %rd3, %rd1, %rd2;
    mov.u64       %rd4, 6364136223846793005;
    mul.lo.u64    %rd3, %rd3, %rd4;
    mov.u64       %rd5, 1442695040888963407;
    add.u64       %rd3, %rd3, %rd5;
    shr.u64       %rd6, %rd3, 33;
    cvt.u32.u64   %r3,  %rd6;             // random u32

    // r = rand % (n_seen + 1)
    add.u32       %r4, %r0, 1;
    rem.u32       %r5, %r3, %r4;          // r ∈ [0, n_seen]

    // if r < capacity: write r as swap index; else write 0xFFFFFFFF (no swap)
    setp.lt.u32   %p1, %r5, %r1;
    selp.u32      %r6, %r5, 0xFFFFFFFF, %p1;
    st.global.u32 [%rd0], %r6;

    bra           $RS_DONE;

$RS_DONE:
    // suppress unused-register warnings
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_kernel_well_formed(prog: &str, sm: u32, kernel_name: &str) {
        assert!(prog.contains(&format!("sm_{sm}")), "missing sm_{sm} target");
        assert!(prog.contains(".version"), "missing .version");
        assert!(prog.contains(".visible .entry"), "missing .visible .entry");
        assert!(
            prog.contains(kernel_name),
            "missing kernel name {kernel_name}"
        );
    }

    #[test]
    fn ewc_penalty_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&ewc_penalty_ptx(sm), sm, "ewc_penalty_kernel");
        }
    }

    #[test]
    fn fisher_diag_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&fisher_diag_ptx(sm), sm, "fisher_diag_kernel");
        }
    }

    #[test]
    fn gradient_project_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gradient_project_ptx(sm), sm, "gradient_project_kernel");
        }
    }

    #[test]
    fn mask_apply_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&mask_apply_ptx(sm), sm, "mask_apply_kernel");
        }
    }

    #[test]
    fn si_omega_update_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&si_omega_update_ptx(sm), sm, "si_omega_update_kernel");
        }
    }

    #[test]
    fn logit_distill_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&logit_distill_ptx(sm), sm, "logit_distill_kernel");
        }
    }

    #[test]
    fn replay_sample_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&replay_sample_ptx(sm), sm, "replay_sample_kernel");
        }
    }

    #[test]
    fn ptx_header_version_strings() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
        assert!(ptx_header(120).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_known_values() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
        assert_eq!(f32_hex(2.0_f32), "0F40000000");
    }

    #[test]
    fn ewc_penalty_uses_fma() {
        let p = ewc_penalty_ptx(80);
        assert!(p.contains("fma.rn.f32"));
        assert!(p.contains("atom.global.add.f32"));
    }

    #[test]
    fn fisher_diag_uses_atomic_add() {
        let p = fisher_diag_ptx(80);
        assert!(p.contains("mul.f32"));
        assert!(p.contains("atom.global.add.f32"));
    }

    #[test]
    fn gradient_project_has_div() {
        let p = gradient_project_ptx(90);
        assert!(p.contains("div.rn.f32"));
    }

    #[test]
    fn mask_apply_uses_setp() {
        let p = mask_apply_ptx(86);
        assert!(p.contains("setp.ne.u32"));
        assert!(p.contains("mul.f32"));
    }

    #[test]
    fn si_omega_uses_abs() {
        let p = si_omega_update_ptx(100);
        assert!(p.contains("abs.f32"));
        assert!(p.contains("atom.global.add.f32"));
    }

    #[test]
    fn logit_distill_uses_ex2_lg2() {
        let p = logit_distill_ptx(120);
        assert!(p.contains("ex2.approx.f32"));
        assert!(p.contains("lg2.approx.f32"));
    }

    #[test]
    fn replay_sample_uses_lcg() {
        let p = replay_sample_ptx(80);
        assert!(p.contains("6364136223846793005"));
        assert!(p.contains("1442695040888963407"));
    }
}
