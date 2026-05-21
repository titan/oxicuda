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

#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

#[must_use]
pub fn bt_reward_loss_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{header}.visible .entry bt_reward_loss_kernel(
    .param .u64 param_chosen,
    .param .u64 param_rejected,
    .param .u64 param_out,
    .param .u32 param_n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_chosen];
    ld.param.u64 %rd1, [param_rejected];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0,  [param_n];

    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r1, %r5;
    mov.u32 %r7, %r4;

$BT_LOOP:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $BT_DONE;

    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // diff = r_w - r_l; bt_loss = -log(sigma(diff)) = log(1 + exp(-diff))
    sub.f32 %f2, %f0, %f1;
    neg.f32 %f3, %f2;
    ex2.approx.f32 %f4, %f3;
    // approx: log1p(exp(-diff)) via 1 + exp(-diff) then lg2 * ln2
    mov.f32 %f5, {ONE};
    add.f32 %f6, %f4, %f5;
    lg2.approx.f32 %f7, %f6;
    mov.f32 %f8, {LN2};
    mul.f32 %f9, %f7, %f8;

    atom.global.add.f32 %f10, [%rd2], %f9;

    add.u32 %r7, %r7, %r6;
    bra $BT_LOOP;

$BT_DONE:
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.f32 %f11, {ZERO};
    mov.u64 %rd6, 0;
    mov.u64 %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        LN2 = f32_hex(std::f32::consts::LN_2),
    )
}

#[must_use]
pub fn dpo_loss_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{header}.visible .entry dpo_loss_kernel(
    .param .u64 param_chosen_lp,
    .param .u64 param_ref_chosen_lp,
    .param .u64 param_rejected_lp,
    .param .u64 param_ref_rejected_lp,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .f32 param_beta
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_chosen_lp];
    ld.param.u64 %rd1, [param_ref_chosen_lp];
    ld.param.u64 %rd2, [param_rejected_lp];
    ld.param.u64 %rd3, [param_ref_rejected_lp];
    ld.param.u64 %rd4, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.f32 %f0,  [param_beta];

    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r1, %r5;
    mov.u32 %r7, %r4;

$DPO_LOOP:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $DPO_DONE;

    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd0, %rd5;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd8, %rd2, %rd5;
    add.u64 %rd9, %rd3, %rd5;

    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd8];
    ld.global.f32 %f4, [%rd9];

    // logit = beta * ((lp_w - ref_w) - (lp_l - ref_l))
    sub.f32 %f5, %f1, %f2;
    sub.f32 %f6, %f3, %f4;
    sub.f32 %f7, %f5, %f6;
    mul.f32 %f8, %f0, %f7;

    // -log_sigmoid(logit) = log(1 + exp(-logit))
    neg.f32 %f9, %f8;
    ex2.approx.f32 %f10, %f9;
    mov.f32 %f11, {ONE};
    add.f32 %f12, %f10, %f11;
    lg2.approx.f32 %f13, %f12;
    mov.f32 %f14, {LN2};
    mul.f32 %f15, %f13, %f14;

    atom.global.add.f32 %f0, [%rd4], %f15;

    add.u32 %r7, %r7, %r6;
    bra $DPO_LOOP;

$DPO_DONE:
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.f32 %f0, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        LN2 = f32_hex(std::f32::consts::LN_2),
    )
}

#[must_use]
pub fn ipo_loss_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{header}.visible .entry ipo_loss_kernel(
    .param .u64 param_chosen_lp,
    .param .u64 param_ref_chosen_lp,
    .param .u64 param_rejected_lp,
    .param .u64 param_ref_rejected_lp,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .f32 param_beta
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_chosen_lp];
    ld.param.u64 %rd1, [param_ref_chosen_lp];
    ld.param.u64 %rd2, [param_rejected_lp];
    ld.param.u64 %rd3, [param_ref_rejected_lp];
    ld.param.u64 %rd4, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.f32 %f0,  [param_beta];

    // target = 1/(2*beta)
    mov.f32 %f1, {TWO};
    mul.f32 %f2, %f0, %f1;
    rcp.approx.f32 %f3, %f2;

    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r1, %r5;
    mov.u32 %r7, %r4;

$IPO_LOOP:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $IPO_DONE;

    mul.wide.u32 %rd5, %r7, 4;
    add.u64 %rd6, %rd0, %rd5;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd8, %rd2, %rd5;
    add.u64 %rd9, %rd3, %rd5;

    ld.global.f32 %f4,  [%rd6];
    ld.global.f32 %f5,  [%rd7];
    ld.global.f32 %f6,  [%rd8];
    ld.global.f32 %f7,  [%rd9];

    // h = (lp_w - ref_w) - (lp_l - ref_l)
    sub.f32 %f8,  %f4, %f5;
    sub.f32 %f9,  %f6, %f7;
    sub.f32 %f10, %f8, %f9;

    // loss = (h - target)^2
    sub.f32 %f11, %f10, %f3;
    mul.f32 %f12, %f11, %f11;

    atom.global.add.f32 %f13, [%rd4], %f12;

    add.u32 %r7, %r7, %r6;
    bra $IPO_LOOP;

$IPO_DONE:
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.f32 %f0, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        TWO = f32_hex(2.0_f32),
    )
}

#[must_use]
pub fn kto_loss_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let z0 = f32_hex(std::f32::consts::LN_2);
    format!(
        r#"{header}.visible .entry kto_loss_kernel(
    .param .u64 param_rewards,
    .param .u64 param_out,
    .param .u32 param_n,
    .param .f32 param_beta,
    .param .u32 param_desirable
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<12>;
    .reg .f32  %f<14>;
    .reg .pred %p0, %p1;

    ld.param.u64 %rd0, [param_rewards];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0,  [param_n];
    ld.param.f32 %f0,  [param_beta];
    ld.param.u32 %r1,  [param_desirable];

    // z0 = ln(2)
    mov.f32 %f1, {Z0};
    // is_desirable = (param_desirable != 0)
    setp.ne.u32 %p1, %r1, 0;

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mov.u32 %r6, %nctaid.x;
    mul.lo.u32 %r7, %r2, %r6;
    mov.u32 %r8, %r5;

$KTO_LOOP:
    setp.ge.u32 %p0, %r8, %r0;
    @%p0 bra $KTO_DONE;

    mul.wide.u32 %rd2, %r8, 4;
    add.u64 %rd3, %rd0, %rd2;
    ld.global.f32 %f2, [%rd3];

    // desirable: arg = beta*(r - z0); undesirable: arg = beta*(z0 - r)
    sub.f32 %f3, %f2, %f1;
    sub.f32 %f4, %f1, %f2;
    selp.f32 %f5, %f3, %f4, %p1;
    mul.f32 %f6, %f0, %f5;

    // sigma(arg) = 1 / (1 + exp(-arg))
    neg.f32 %f7, %f6;
    ex2.approx.f32 %f8, %f7;
    mov.f32 %f9, {ONE};
    add.f32 %f10, %f8, %f9;
    rcp.approx.f32 %f11, %f10;

    // loss_i = 1 - sigma(arg)
    sub.f32 %f12, %f9, %f11;

    atom.global.add.f32 %f13, [%rd1], %f12;

    add.u32 %r8, %r8, %r7;
    bra $KTO_LOOP;

$KTO_DONE:
    mov.u32 %r9,  0;
    mov.u32 %r10, 0;
    mov.u32 %r11, 0;
    mov.f32 %f0, {ZERO};
    mov.u64 %rd4, 0;
    mov.u64 %rd5, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        Z0 = z0,
    )
}

#[must_use]
pub fn orpo_odds_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-7_f32);
    format!(
        r#"{header}.visible .entry orpo_odds_kernel(
    .param .u64 param_logps,
    .param .u64 param_out,
    .param .u32 param_n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_logps];
    ld.param.u64 %rd1, [param_out];
    ld.param.u32 %r0,  [param_n];

    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r1, %r5;
    mov.u32 %r7, %r4;

$ORPO_LOOP:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $ORPO_DONE;

    mul.wide.u32 %rd2, %r7, 4;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.f32 %f0, [%rd3];

    // p = exp(lp); odds = p / (1 - p + eps); log_odds = log(odds)
    // Use ex2: exp(x) = ex2(x * log2e)
    mov.f32 %f1, {LOG2E};
    mul.f32 %f2, %f0, %f1;
    ex2.approx.f32 %f3, %f2;

    // 1 - p + eps
    mov.f32 %f4, {ONE};
    sub.f32 %f5, %f4, %f3;
    mov.f32 %f6, {EPS};
    add.f32 %f7, %f5, %f6;

    // odds = p / (1 - p + eps)
    div.rn.f32 %f8, %f3, %f7;

    // log_odds = ln(odds) = log2(odds) * ln2
    lg2.approx.f32 %f9, %f8;
    mov.f32 %f10, {LN2};
    mul.f32 %f11, %f9, %f10;

    st.global.f32 [%rd4], %f11;

    add.u32 %r7, %r7, %r6;
    bra $ORPO_LOOP;

$ORPO_DONE:
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.f32 %f0, {ZERO};
    mov.u64 %rd5, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
        ONE = f32_hex(1.0_f32),
        LOG2E = f32_hex(std::f32::consts::LOG2_E),
        LN2 = f32_hex(std::f32::consts::LN_2),
    )
}

#[must_use]
pub fn rlhf_kl_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{header}.visible .entry rlhf_kl_kernel(
    .param .u64 param_log_probs,
    .param .u64 param_ref_log_probs,
    .param .u64 param_out,
    .param .u32 param_n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [param_log_probs];
    ld.param.u64 %rd1, [param_ref_log_probs];
    ld.param.u64 %rd2, [param_out];
    ld.param.u32 %r0,  [param_n];

    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mov.u32 %r5, %nctaid.x;
    mul.lo.u32 %r6, %r1, %r5;
    mov.u32 %r7, %r4;

$KL_LOOP:
    setp.ge.u32 %p0, %r7, %r0;
    @%p0 bra $KL_DONE;

    mul.wide.u32 %rd3, %r7, 4;
    add.u64 %rd4, %rd0, %rd3;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // forward KL approx: exp(lp) * (lp - ref_lp)
    sub.f32 %f2, %f0, %f1;
    mov.f32 %f3, {LOG2E};
    mul.f32 %f4, %f0, %f3;
    ex2.approx.f32 %f5, %f4;
    mul.f32 %f6, %f5, %f2;

    st.global.f32 [%rd6], %f6;

    add.u32 %r7, %r7, %r6;
    bra $KL_LOOP;

$KL_DONE:
    mov.u32 %r8, 0;
    mov.u32 %r9, 0;
    mov.f32 %f7, {ZERO};
    mov.f32 %f8, {ZERO};
    mov.f32 %f9, {ZERO};
    mov.u64 %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
        LOG2E = f32_hex(std::f32::consts::LOG2_E),
    )
}

#[must_use]
pub fn sft_mask_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{header}.visible .entry sft_mask_kernel(
    .param .u64 param_logits,
    .param .u64 param_labels,
    .param .u64 param_mask,
    .param .u64 param_out,
    .param .u32 param_n_tokens,
    .param .u32 param_n_vocab
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<14>;
    .reg .u8   %rc0;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1;

    ld.param.u64 %rd0, [param_logits];
    ld.param.u64 %rd1, [param_labels];
    ld.param.u64 %rd2, [param_mask];
    ld.param.u64 %rd3, [param_out];
    ld.param.u32 %r0,  [param_n_tokens];
    ld.param.u32 %r1,  [param_n_vocab];

    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mov.u32 %r6, %nctaid.x;
    mul.lo.u32 %r7, %r2, %r6;
    mov.u32 %r8, %r5;

$SFT_LOOP:
    setp.ge.u32 %p0, %r8, %r0;
    @%p0 bra $SFT_DONE;

    // Load mask byte
    cvt.u64.u32 %rd4, %r8;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u8 %rc0, [%rd5];
    cvt.u32.u8 %r9, %rc0;
    setp.eq.u32 %p1, %r9, 0;
    @%p1 bra $SFT_NEXT;

    // Load label
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.u32 %r10, [%rd7];

    // Load the label-logit at the row-major offset logits[token * n_vocab + label].
    mul.lo.u32 %r11, %r8, %r1;
    add.u32 %r12, %r11, %r10;
    mul.wide.u32 %rd8, %r12, 4;
    add.u64 %rd9, %rd0, %rd8;
    ld.global.f32 %f0, [%rd9];

    // Write the per-token CE contribution: the negated label-logit. The
    // log-sum-exp softmax denominator over the vocab row is added by the host
    // caller (see `sft::loss::masked_token_ce`), completing the device/host split.
    neg.f32 %f1, %f0;
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd4, %rd3, %rd6;
    st.global.f32 [%rd4], %f1;

$SFT_NEXT:
    add.u32 %r8, %r8, %r7;
    bra $SFT_LOOP;

$SFT_DONE:
    mov.u32 %r13, 0;
    mov.f32 %f2, {ZERO};
    mov.f32 %f3, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

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
    fn bt_reward_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&bt_reward_loss_ptx(sm), sm, "bt_reward_loss_kernel");
        }
    }

    #[test]
    fn dpo_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&dpo_loss_ptx(sm), sm, "dpo_loss_kernel");
        }
    }

    #[test]
    fn ipo_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&ipo_loss_ptx(sm), sm, "ipo_loss_kernel");
        }
    }

    #[test]
    fn kto_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&kto_loss_ptx(sm), sm, "kto_loss_kernel");
        }
    }

    #[test]
    fn orpo_odds_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&orpo_odds_ptx(sm), sm, "orpo_odds_kernel");
        }
    }

    #[test]
    fn rlhf_kl_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&rlhf_kl_ptx(sm), sm, "rlhf_kl_kernel");
        }
    }

    #[test]
    fn sft_mask_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&sft_mask_ptx(sm), sm, "sft_mask_kernel");
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
}
