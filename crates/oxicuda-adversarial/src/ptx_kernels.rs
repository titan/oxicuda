//! PTX GPU kernel sources for adversarial robustness operations.
//!
//! Each function returns a PTX program as a `String` for runtime compilation.
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`fgsm_step_ptx`] | `x_adv = x + ε · sign(grad)`, clamped to `[lo, hi]` |
//! | [`pgd_proj_l_inf_ptx`] | L∞ projection: clamp `x` to `x_orig ± ε` then `[lo, hi]` |
//! | [`pgd_proj_l2_ptx`] | L2 projection: scale `δ = x − x_orig` so `‖δ‖₂ ≤ ε` |
//! | [`smoothing_noise_ptx`] | Gaussian noise via inline LCG + Box-Muller |
//! | [`grad_sign_ptx`] | `out[i] = sign(grad[i])` (used by FGSM/PGD inner loop) |
//! | [`certified_radius_reduce_ptx`] | Reduces a per-class count vector to `(top, runner_up)` |
//! | [`attack_loss_grad_ptx`] | Per-element scale-by-loss-grad-direction step |

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

// ─── Kernel 1: fgsm_step ─────────────────────────────────────────────────────

/// `x_adv[i] = clamp(x[i] + eps * sign(grad[i]), lo, hi)` element-wise.
#[must_use]
pub fn fgsm_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let neg_one = f32_hex(-1.0_f32);
    format!(
        r#"{hdr}// fgsm_step_kernel: x_adv = clamp(x + eps * sign(grad), lo, hi)
.visible .entry fgsm_step_kernel(
    .param .u64 p_x,
    .param .u64 p_grad,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 eps,
    .param .f32 lo,
    .param .f32 hi
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_grad];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [eps];
    ld.param.f32  %f1,  [lo];
    ld.param.f32  %f2,  [hi];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // tid global
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride
    mov.u32       %r7, %r4;

$FGSM_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $FGSM_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];           // x[i]
    ld.global.f32 %f4, [%rd5];           // grad[i]

    // sign(grad)
    setp.gt.f32   %p0, %f4, {ZERO};
    setp.lt.f32   %p1, %f4, {ZERO};
    selp.f32      %f5, {ONE}, {ZERO}, %p0;
    selp.f32      %f6, {NEG_ONE}, %f5, %p1;

    // x + eps * sign(grad)
    fma.rn.f32    %f7, %f0, %f6, %f3;

    // clamp(lo, hi)
    max.f32       %f8, %f7, %f1;
    min.f32       %f9, %f8, %f2;

    add.u64       %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f9;

    add.u32       %r7, %r7, %r6;
    bra           $FGSM_LOOP;

$FGSM_DONE:
    // Suppress unused-register warnings.
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f10, {ZERO};
    mov.f32       %f11, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        NEG_ONE = neg_one,
    )
}

// ─── Kernel 2: pgd_proj_l_inf ────────────────────────────────────────────────

/// L∞ projection: `out[i] = clamp(clamp(x[i], x_orig[i]-eps, x_orig[i]+eps), lo, hi)`.
#[must_use]
pub fn pgd_proj_l_inf_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// pgd_proj_l_inf_kernel: clamp x to ε-ball around x_orig and into [lo, hi].
.visible .entry pgd_proj_l_inf_kernel(
    .param .u64 p_x,
    .param .u64 p_orig,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 eps,
    .param .f32 lo,
    .param .f32 hi
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_orig];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [eps];
    ld.param.f32  %f1,  [lo];
    ld.param.f32  %f2,  [hi];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$PGD_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $PGD_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];           // x[i]
    ld.global.f32 %f4, [%rd5];           // x_orig[i]

    sub.f32       %f5, %f4, %f0;         // x_orig - eps
    add.f32       %f6, %f4, %f0;         // x_orig + eps
    max.f32       %f7, %f3, %f5;
    min.f32       %f7, %f7, %f6;

    // outer clamp to [lo, hi]
    max.f32       %f8, %f7, %f1;
    min.f32       %f9, %f8, %f2;

    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f9;

    add.u32       %r7, %r7, %r6;
    bra           $PGD_LOOP;

$PGD_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#
    )
}

// ─── Kernel 3: pgd_proj_l2 ───────────────────────────────────────────────────

/// L2 projection: scales `δ = x − x_orig` so its L2 norm is at most `eps`.
/// The norm is supplied by the host (computed via dedicated reduction kernel).
#[must_use]
pub fn pgd_proj_l2_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// pgd_proj_l2_kernel: scale (x − x_orig) so ‖δ‖₂ ≤ eps.
.visible .entry pgd_proj_l2_kernel(
    .param .u64 p_x,
    .param .u64 p_orig,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 eps,
    .param .f32 norm,
    .param .f32 lo,
    .param .f32 hi
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_orig];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [eps];
    ld.param.f32  %f1,  [norm];
    ld.param.f32  %f2,  [lo];
    ld.param.f32  %f3,  [hi];

    // factor = min(1, eps / norm)
    setp.gt.f32   %p1, %f1, %f0;
    div.rn.f32    %f4, %f0, %f1;
    selp.f32      %f5, %f4, {ONE}, %p1;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$L2_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $L2_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f6, [%rd4];           // x[i]
    ld.global.f32 %f7, [%rd5];           // x_orig[i]

    sub.f32       %f8, %f6, %f7;         // delta
    fma.rn.f32    %f9, %f8, %f5, %f7;    // x_orig + factor * delta

    max.f32       %f10, %f9, %f2;
    min.f32       %f11, %f10, %f3;

    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f11;

    add.u32       %r7, %r7, %r6;
    bra           $L2_LOOP;

$L2_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ONE = one,
    )
}

// ─── Kernel 4: smoothing_noise ───────────────────────────────────────────────

/// Add Gaussian noise ε ~ N(0, σ²) to `x` element-wise via inline LCG +
/// Box-Muller. Used by randomized smoothing certification.
#[must_use]
pub fn smoothing_noise_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let two = f32_hex(2.0_f32);
    let pi = f32_hex(std::f32::consts::PI);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    format!(
        r#"{hdr}// smoothing_noise_kernel: out[i] = x[i] + sigma * z, z ~ N(0, 1)
.visible .entry smoothing_noise_kernel(
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 sigma,
    .param .u64 seed
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<14>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [sigma];
    ld.param.u64  %rd2, [seed];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$SN_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $SN_DONE;

    // Two independent uniforms via two LCG steps, mixed with i.
    cvt.u64.u32   %rd3, %r7;
    xor.b64       %rd4, %rd2, %rd3;
    mov.u64       %rd5, 6364136223846793005;
    mul.lo.u64    %rd4, %rd4, %rd5;
    mov.u64       %rd6, 1442695040888963407;
    add.u64       %rd4, %rd4, %rd6;
    shr.u64       %rd7, %rd4, 33;
    cvt.u32.u64   %r8,  %rd7;
    cvt.rn.f32.u32 %f1, %r8;
    mul.f32        %f1, %f1, 0F2F800000;  // / 2^32
    // u1 = max(eps, f1)
    max.f32        %f1, %f1, 0F2F800000;  // avoid log(0)

    // Second uniform from another LCG step.
    add.u64       %rd4, %rd4, %rd5;
    shr.u64       %rd7, %rd4, 33;
    cvt.u32.u64   %r9,  %rd7;
    cvt.rn.f32.u32 %f2, %r9;
    mul.f32        %f2, %f2, 0F2F800000;

    // Box-Muller: z = sqrt(-2 ln u1) * cos(2π u2)
    // ln(u1) = lg2(u1) / log2(e)
    lg2.approx.f32 %f3, %f1;
    div.rn.f32     %f3, %f3, {LOG2E};      // natural log
    mul.f32        %f4, %f3, {TWO};
    neg.f32        %f4, %f4;               // -2 ln u1
    sqrt.approx.f32 %f5, %f4;              // r
    mul.f32        %f6, %f2, {TWO};
    mul.f32        %f6, %f6, {PI};         // 2π u2
    cos.approx.f32 %f7, %f6;
    mul.f32        %f8, %f5, %f7;          // z

    // x + sigma * z
    mul.wide.u32   %rd3, %r7, 4;
    add.u64        %rd4, %rd0, %rd3;
    ld.global.f32  %f9, [%rd4];
    fma.rn.f32     %f10, %f0, %f8, %f9;

    add.u64        %rd5, %rd1, %rd3;
    st.global.f32  [%rd5], %f10;

    add.u32        %r7, %r7, %r6;
    bra            $SN_LOOP;

$SN_DONE:
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.u32       %r12, 0;
    mov.u32       %r13, 0;
    mov.f32       %f11, 0F00000000;
    mov.f32       %f12, 0F00000000;
    mov.f32       %f13, 0F00000000;
    ret;
}}
"#,
        TWO = two,
        PI = pi,
        LOG2E = log2e,
    )
}

// ─── Kernel 5: grad_sign ─────────────────────────────────────────────────────

/// `out[i] = sign(grad[i])` — `+1, 0, -1` per element.
#[must_use]
pub fn grad_sign_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let neg_one = f32_hex(-1.0_f32);
    format!(
        r#"{hdr}// grad_sign_kernel: out[i] = sign(grad[i])
.visible .entry grad_sign_kernel(
    .param .u64 p_grad,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_grad];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$GS_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $GS_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

    setp.gt.f32   %p1, %f0, {ZERO};
    setp.lt.f32   %p2, %f0, {ZERO};
    selp.f32      %f1, {ONE}, {ZERO}, %p1;
    selp.f32      %f2, {NEG_ONE}, %f1, %p2;

    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f2;

    add.u32       %r7, %r7, %r6;
    bra           $GS_LOOP;

$GS_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd5, 0;
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        NEG_ONE = neg_one,
    )
}

// ─── Kernel 6: certified_radius_reduce ───────────────────────────────────────

/// Reduces a per-class count vector `[K]` to the index of the top class
/// (one block per query). Used to read off the smoothed predictor's argmax.
#[must_use]
pub fn certified_radius_reduce_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// certified_radius_reduce_kernel: argmax of [K] count vector.
.visible .entry certified_radius_reduce_kernel(
    .param .u64 p_counts,
    .param .u64 p_argmax,
    .param .u32 k_classes
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<14>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_counts];
    ld.param.u64  %rd1, [p_argmax];
    ld.param.u32  %r0,  [k_classes];

    mov.u32       %r2, %tid.x;
    setp.ne.u32   %p0, %r2, 0;
    @%p0 bra $CR_DONE;

    // Single-thread argmax (per block); production version would use a
    // warp-shuffle reduce.
    mov.u32       %r3, 0;                 // best idx
    ld.global.u32 %r4, [%rd0];            // best count
    mov.u32       %r5, 1;
$CR_LOOP:
    setp.ge.u32   %p1, %r5, %r0;
    @%p1 bra $CR_WRITE;

    mul.wide.u32  %rd2, %r5, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.u32 %r6, [%rd3];
    setp.gt.u32   %p1, %r6, %r4;
    selp.b32      %r4, %r6, %r4, %p1;
    selp.b32      %r3, %r5, %r3, %p1;
    add.u32       %r5, %r5, 1;
    bra           $CR_LOOP;

$CR_WRITE:
    mov.u32       %r7, %ctaid.x;
    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r3;

$CR_DONE:
    // Suppress unused-register warnings.
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.u32       %r12, 0;
    mov.u32       %r13, 0;
    ret;
}}
"#
    )
}

// ─── Kernel 7: attack_loss_grad ──────────────────────────────────────────────

/// `out[i] = x[i] + alpha * direction[i]` where direction is host-normalised.
/// Used as the inner loop of MIM/PGD with momentum-accumulated gradient.
#[must_use]
pub fn attack_loss_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// attack_loss_grad_kernel: out[i] = clamp(x[i] + alpha * dir[i], lo, hi)
.visible .entry attack_loss_grad_kernel(
    .param .u64 p_x,
    .param .u64 p_dir,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 alpha,
    .param .f32 lo,
    .param .f32 hi
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_dir];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [alpha];
    ld.param.f32  %f1,  [lo];
    ld.param.f32  %f2,  [hi];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$ALG_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $ALG_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];
    ld.global.f32 %f4, [%rd5];
    fma.rn.f32    %f5, %f0, %f4, %f3;
    max.f32       %f6, %f5, %f1;
    min.f32       %f7, %f6, %f2;

    add.u64       %rd3, %rd2, %rd3;
    st.global.f32 [%rd3], %f7;

    add.u32       %r7, %r7, %r6;
    bra           $ALG_LOOP;

$ALG_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f8, 0F00000000;
    mov.f32       %f9, 0F00000000;
    ret;
}}
"#
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_kernel_well_formed(prog: &str, sm: u32, kernel_name: &str) {
        assert!(prog.contains(&format!("sm_{sm}")));
        assert!(prog.contains(".version"));
        assert!(prog.contains(".visible .entry"));
        assert!(prog.contains(kernel_name));
    }

    #[test]
    fn fgsm_step_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&fgsm_step_ptx(sm), sm, "fgsm_step_kernel");
        }
    }

    #[test]
    fn pgd_proj_l_inf_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&pgd_proj_l_inf_ptx(sm), sm, "pgd_proj_l_inf_kernel");
        }
    }

    #[test]
    fn pgd_proj_l2_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&pgd_proj_l2_ptx(sm), sm, "pgd_proj_l2_kernel");
        }
    }

    #[test]
    fn smoothing_noise_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&smoothing_noise_ptx(sm), sm, "smoothing_noise_kernel");
        }
    }

    #[test]
    fn grad_sign_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&grad_sign_ptx(sm), sm, "grad_sign_kernel");
        }
    }

    #[test]
    fn certified_radius_reduce_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(
                &certified_radius_reduce_ptx(sm),
                sm,
                "certified_radius_reduce_kernel",
            );
        }
    }

    #[test]
    fn attack_loss_grad_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&attack_loss_grad_ptx(sm), sm, "attack_loss_grad_kernel");
        }
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_sanity() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn fgsm_uses_fma() {
        assert!(fgsm_step_ptx(80).contains("fma.rn.f32"));
    }

    #[test]
    fn pgd_l2_uses_div() {
        assert!(pgd_proj_l2_ptx(80).contains("div.rn.f32"));
    }
}
