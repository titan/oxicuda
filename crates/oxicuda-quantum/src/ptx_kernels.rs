//! PTX kernel string generators for quantum simulation on GPU.

use crate::handle::SmVersion;

fn ptx_header(sm: u32) -> String {
    let sv = SmVersion(sm);
    format!(
        ".version {}\n.target {}\n.address_size 64\n",
        sv.ptx_version_str(),
        sv.target_str()
    )
}

/// Single-qubit gate kernel: applies a 2×2 complex gate to amplitude pairs.
pub fn statevec_apply_1q_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let inv_sqrt2_hex = format!("0F{:08X}", (std::f32::consts::FRAC_1_SQRT_2).to_bits());
    format!(
        r#"{header}
// statevec_apply_1q: applies 2x2 complex gate to amplitude pairs indexed by qubit mask
// Parameters: amp_re *f32, amp_im *f32, mask u32, g00re g00im g01re g01im g10re g10im g11re g11im f32x8, n_pairs u32
// inv_sqrt2 = {inv_sqrt2_hex}
.visible .entry statevec_apply_1q(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_mask,
    .param .f32 param_g00re,
    .param .f32 param_g00im,
    .param .f32 param_g01re,
    .param .f32 param_g01im,
    .param .f32 param_g10re,
    .param .f32 param_g10im,
    .param .f32 param_g11re,
    .param .f32 param_g11im,
    .param .u32 param_n_pairs
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<32>;
    .reg .pred %p0;

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    // stride loop guard
    ld.param.u32 %r4, [param_n_pairs];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_mask];

    // i0 = ((tid & ~(mask-1)) << 1) | (tid & (mask-1))  — insert a 0 bit at the qubit slot
    sub.u32 %r8, %r5, 1;       // mask - 1 (low-bit mask)
    not.b32 %r6, %r8;          // ~(mask-1) (high-bit mask)
    and.b32 %r7, %r3, %r6;
    shl.b32 %r7, %r7, 1;
    and.b32 %r9, %r3, %r8;
    or.b32 %r10, %r7, %r9;

    // i1 = i0 | mask
    or.b32 %r11, %r10, %r5;

    // load base pointers
    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    // byte offsets: i0*4, i1*4
    mul.lo.u32 %r12, %r10, 4;
    mul.lo.u32 %r13, %r11, 4;

    // convert to u64 offsets
    cvt.u64.u32 %rd2, %r12;
    cvt.u64.u32 %rd3, %r13;

    // addresses for re and im at i0 and i1
    add.u64 %rd4, %rd0, %rd2;
    add.u64 %rd5, %rd0, %rd3;
    add.u64 %rd6, %rd1, %rd2;
    add.u64 %rd7, %rd1, %rd3;

    // load x0 = amp[i0] (complex), x1 = amp[i1] (complex)
    ld.global.f32 %f0, [%rd4];   // x0.re
    ld.global.f32 %f1, [%rd6];   // x0.im
    ld.global.f32 %f2, [%rd5];   // x1.re
    ld.global.f32 %f3, [%rd7];   // x1.im

    // load gate elements
    ld.param.f32 %f4,  [param_g00re];
    ld.param.f32 %f5,  [param_g00im];
    ld.param.f32 %f6,  [param_g01re];
    ld.param.f32 %f7,  [param_g01im];
    ld.param.f32 %f8,  [param_g10re];
    ld.param.f32 %f9,  [param_g10im];
    ld.param.f32 %f10, [param_g11re];
    ld.param.f32 %f11, [param_g11im];

    // a0 = g00*x0 + g01*x1 (complex multiply-add)
    // re(a0) = g00re*x0re - g00im*x0im + g01re*x1re - g01im*x1im
    mul.f32 %f12, %f4,  %f0;
    fma.rn.f32 %f12, %f5,  %f1, %f12;  // -= g00im*x0im  (negate via sub below)
    // use: re(a0) = g00re*x0re - g00im*x0im + g01re*x1re - g01im*x1im
    mul.f32 %f13, %f4,  %f0;
    mul.f32 %f14, %f5,  %f1;
    sub.f32 %f15, %f13, %f14;           // g00re*x0re - g00im*x0im
    mul.f32 %f13, %f6,  %f2;
    mul.f32 %f14, %f7,  %f3;
    sub.f32 %f16, %f13, %f14;           // g01re*x1re - g01im*x1im
    add.f32 %f17, %f15, %f16;           // re(a0)

    // im(a0) = g00re*x0im + g00im*x0re + g01re*x1im + g01im*x1re
    mul.f32 %f13, %f4,  %f1;
    mul.f32 %f14, %f5,  %f0;
    add.f32 %f15, %f13, %f14;
    mul.f32 %f13, %f6,  %f3;
    mul.f32 %f14, %f7,  %f2;
    add.f32 %f16, %f13, %f14;
    add.f32 %f18, %f15, %f16;           // im(a0)

    // a1 = g10*x0 + g11*x1
    mul.f32 %f13, %f8,  %f0;
    mul.f32 %f14, %f9,  %f1;
    sub.f32 %f15, %f13, %f14;
    mul.f32 %f13, %f10, %f2;
    mul.f32 %f14, %f11, %f3;
    sub.f32 %f16, %f13, %f14;
    add.f32 %f19, %f15, %f16;           // re(a1)

    mul.f32 %f13, %f8,  %f1;
    mul.f32 %f14, %f9,  %f0;
    add.f32 %f15, %f13, %f14;
    mul.f32 %f13, %f10, %f3;
    mul.f32 %f14, %f11, %f2;
    add.f32 %f16, %f13, %f14;
    add.f32 %f20, %f15, %f16;           // im(a1)

    // store results
    st.global.f32 [%rd4], %f17;
    st.global.f32 [%rd6], %f18;
    st.global.f32 [%rd5], %f19;
    st.global.f32 [%rd7], %f20;

DONE:
    ret;
}}
"#
    )
}

/// Two-qubit gate kernel: applies a 4×4 complex gate to groups of 4 amplitudes.
pub fn statevec_apply_2q_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    format!(
        r#"{header}
// statevec_apply_2q: applies 4x4 complex gate; one thread per group of 4 amplitudes
// Parameters: amp_re *f32, amp_im *f32, mask0 u32, mask1 u32, n_groups u32, gate_re[16] *f32, gate_im[16] *f32
.visible .entry statevec_apply_2q(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_mask0,
    .param .u32 param_mask1,
    .param .u32 param_n_groups,
    .param .u64 param_gate_re,
    .param .u64 param_gate_im
)
{{
    .reg .u32 %r<32>;
    .reg .u64 %rd<16>;
    .reg .f32 %f<64>;
    .reg .pred %p0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n_groups];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_mask0];
    ld.param.u32 %r6, [param_mask1];

    // Reconstruct the 4 group indices by inserting two 0 bits into tid at the two
    // qubit positions. lo = lower mask, hi = higher mask (single-bit masks).
    min.u32 %r7, %r5, %r6;
    max.u32 %r8, %r5, %r6;

    // step 1: insert a 0 bit at the lower qubit slot
    sub.u32 %r9, %r7, 1;        // lo - 1
    not.b32 %r10, %r9;          // ~(lo-1)
    and.b32 %r15, %r3, %r10;
    shl.b32 %r15, %r15, 1;
    and.b32 %r16, %r3, %r9;
    or.b32  %r17, %r15, %r16;   // tid with a 0 inserted below lo

    // step 2: insert a 0 bit at the higher qubit slot. Because we inserted the
    // lower hole first (low -> high), the higher hole's final position is just hi.
    sub.u32 %r19, %r8, 1;       // hi - 1
    not.b32 %r10, %r19;         // ~(hi-1)
    and.b32 %r15, %r17, %r10;
    shl.b32 %r15, %r15, 1;
    and.b32 %r16, %r17, %r19;
    or.b32  %r11, %r15, %r16;   // i00 (both qubit bits = 0)

    or.b32  %r12, %r11, %r6;    // i01 = i00 | mask1
    or.b32  %r13, %r11, %r5;    // i10 = i00 | mask0
    or.b32  %r14, %r11, %r5;
    or.b32  %r14, %r14, %r6;    // i11 = i00 | mask0 | mask1

    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];
    ld.param.u64 %rd2, [param_gate_re];
    ld.param.u64 %rd3, [param_gate_im];

    // load 4 input amplitudes (re: f0,f2,f4,f6 ; im: f1,f3,f5,f7)
    mul.lo.u32 %r20, %r11, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    ld.global.f32 %f0, [%rd8];    // x0.re
    ld.global.f32 %f1, [%rd12];   // x0.im

    mul.lo.u32 %r20, %r12, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    ld.global.f32 %f2, [%rd8];    // x1.re
    ld.global.f32 %f3, [%rd12];   // x1.im

    mul.lo.u32 %r20, %r13, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    ld.global.f32 %f4, [%rd8];    // x2.re
    ld.global.f32 %f5, [%rd12];   // x2.im

    mul.lo.u32 %r20, %r14, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    ld.global.f32 %f6, [%rd8];    // x3.re
    ld.global.f32 %f7, [%rd12];   // x3.im

    // 4x4 complex matrix-vector product: y[j] = sum_k gate[j,k] * x[k].
    // gate_re[j*4+k] / gate_im[j*4+k] at byte offset (j*4+k)*4.
    // Row 0 -> f16(re),f17(im); Row 1 -> f18,f19; Row 2 -> f20,f21; Row 3 -> f22,f23.

    // ---- Row 0 (gate offsets 0,4,8,12) ----
    mov.f32 %f16, 0F00000000;
    mov.f32 %f17, 0F00000000;
    ld.global.f32 %f8, [%rd2+0];
    ld.global.f32 %f9, [%rd3+0];
    mul.f32 %f10, %f8, %f0;
    mul.f32 %f11, %f9, %f1;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f16, %f16, %f10;
    mul.f32 %f10, %f8, %f1;
    mul.f32 %f11, %f9, %f0;
    add.f32 %f10, %f10, %f11;
    add.f32 %f17, %f17, %f10;
    ld.global.f32 %f8, [%rd2+4];
    ld.global.f32 %f9, [%rd3+4];
    mul.f32 %f10, %f8, %f2;
    mul.f32 %f11, %f9, %f3;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f16, %f16, %f10;
    mul.f32 %f10, %f8, %f3;
    mul.f32 %f11, %f9, %f2;
    add.f32 %f10, %f10, %f11;
    add.f32 %f17, %f17, %f10;
    ld.global.f32 %f8, [%rd2+8];
    ld.global.f32 %f9, [%rd3+8];
    mul.f32 %f10, %f8, %f4;
    mul.f32 %f11, %f9, %f5;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f16, %f16, %f10;
    mul.f32 %f10, %f8, %f5;
    mul.f32 %f11, %f9, %f4;
    add.f32 %f10, %f10, %f11;
    add.f32 %f17, %f17, %f10;
    ld.global.f32 %f8, [%rd2+12];
    ld.global.f32 %f9, [%rd3+12];
    mul.f32 %f10, %f8, %f6;
    mul.f32 %f11, %f9, %f7;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f16, %f16, %f10;
    mul.f32 %f10, %f8, %f7;
    mul.f32 %f11, %f9, %f6;
    add.f32 %f10, %f10, %f11;
    add.f32 %f17, %f17, %f10;

    // ---- Row 1 (gate offsets 16,20,24,28) ----
    mov.f32 %f18, 0F00000000;
    mov.f32 %f19, 0F00000000;
    ld.global.f32 %f8, [%rd2+16];
    ld.global.f32 %f9, [%rd3+16];
    mul.f32 %f10, %f8, %f0;
    mul.f32 %f11, %f9, %f1;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f18, %f18, %f10;
    mul.f32 %f10, %f8, %f1;
    mul.f32 %f11, %f9, %f0;
    add.f32 %f10, %f10, %f11;
    add.f32 %f19, %f19, %f10;
    ld.global.f32 %f8, [%rd2+20];
    ld.global.f32 %f9, [%rd3+20];
    mul.f32 %f10, %f8, %f2;
    mul.f32 %f11, %f9, %f3;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f18, %f18, %f10;
    mul.f32 %f10, %f8, %f3;
    mul.f32 %f11, %f9, %f2;
    add.f32 %f10, %f10, %f11;
    add.f32 %f19, %f19, %f10;
    ld.global.f32 %f8, [%rd2+24];
    ld.global.f32 %f9, [%rd3+24];
    mul.f32 %f10, %f8, %f4;
    mul.f32 %f11, %f9, %f5;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f18, %f18, %f10;
    mul.f32 %f10, %f8, %f5;
    mul.f32 %f11, %f9, %f4;
    add.f32 %f10, %f10, %f11;
    add.f32 %f19, %f19, %f10;
    ld.global.f32 %f8, [%rd2+28];
    ld.global.f32 %f9, [%rd3+28];
    mul.f32 %f10, %f8, %f6;
    mul.f32 %f11, %f9, %f7;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f18, %f18, %f10;
    mul.f32 %f10, %f8, %f7;
    mul.f32 %f11, %f9, %f6;
    add.f32 %f10, %f10, %f11;
    add.f32 %f19, %f19, %f10;

    // ---- Row 2 (gate offsets 32,36,40,44) ----
    mov.f32 %f20, 0F00000000;
    mov.f32 %f21, 0F00000000;
    ld.global.f32 %f8, [%rd2+32];
    ld.global.f32 %f9, [%rd3+32];
    mul.f32 %f10, %f8, %f0;
    mul.f32 %f11, %f9, %f1;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f20, %f20, %f10;
    mul.f32 %f10, %f8, %f1;
    mul.f32 %f11, %f9, %f0;
    add.f32 %f10, %f10, %f11;
    add.f32 %f21, %f21, %f10;
    ld.global.f32 %f8, [%rd2+36];
    ld.global.f32 %f9, [%rd3+36];
    mul.f32 %f10, %f8, %f2;
    mul.f32 %f11, %f9, %f3;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f20, %f20, %f10;
    mul.f32 %f10, %f8, %f3;
    mul.f32 %f11, %f9, %f2;
    add.f32 %f10, %f10, %f11;
    add.f32 %f21, %f21, %f10;
    ld.global.f32 %f8, [%rd2+40];
    ld.global.f32 %f9, [%rd3+40];
    mul.f32 %f10, %f8, %f4;
    mul.f32 %f11, %f9, %f5;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f20, %f20, %f10;
    mul.f32 %f10, %f8, %f5;
    mul.f32 %f11, %f9, %f4;
    add.f32 %f10, %f10, %f11;
    add.f32 %f21, %f21, %f10;
    ld.global.f32 %f8, [%rd2+44];
    ld.global.f32 %f9, [%rd3+44];
    mul.f32 %f10, %f8, %f6;
    mul.f32 %f11, %f9, %f7;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f20, %f20, %f10;
    mul.f32 %f10, %f8, %f7;
    mul.f32 %f11, %f9, %f6;
    add.f32 %f10, %f10, %f11;
    add.f32 %f21, %f21, %f10;

    // ---- Row 3 (gate offsets 48,52,56,60) ----
    mov.f32 %f22, 0F00000000;
    mov.f32 %f23, 0F00000000;
    ld.global.f32 %f8, [%rd2+48];
    ld.global.f32 %f9, [%rd3+48];
    mul.f32 %f10, %f8, %f0;
    mul.f32 %f11, %f9, %f1;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f22, %f22, %f10;
    mul.f32 %f10, %f8, %f1;
    mul.f32 %f11, %f9, %f0;
    add.f32 %f10, %f10, %f11;
    add.f32 %f23, %f23, %f10;
    ld.global.f32 %f8, [%rd2+52];
    ld.global.f32 %f9, [%rd3+52];
    mul.f32 %f10, %f8, %f2;
    mul.f32 %f11, %f9, %f3;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f22, %f22, %f10;
    mul.f32 %f10, %f8, %f3;
    mul.f32 %f11, %f9, %f2;
    add.f32 %f10, %f10, %f11;
    add.f32 %f23, %f23, %f10;
    ld.global.f32 %f8, [%rd2+56];
    ld.global.f32 %f9, [%rd3+56];
    mul.f32 %f10, %f8, %f4;
    mul.f32 %f11, %f9, %f5;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f22, %f22, %f10;
    mul.f32 %f10, %f8, %f5;
    mul.f32 %f11, %f9, %f4;
    add.f32 %f10, %f10, %f11;
    add.f32 %f23, %f23, %f10;
    ld.global.f32 %f8, [%rd2+60];
    ld.global.f32 %f9, [%rd3+60];
    mul.f32 %f10, %f8, %f6;
    mul.f32 %f11, %f9, %f7;
    sub.f32 %f10, %f10, %f11;
    add.f32 %f22, %f22, %f10;
    mul.f32 %f10, %f8, %f7;
    mul.f32 %f11, %f9, %f6;
    add.f32 %f10, %f10, %f11;
    add.f32 %f23, %f23, %f10;

    // store y0..y3 to i00,i01,i10,i11 (all inputs already read, so no race)
    mul.lo.u32 %r20, %r11, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    st.global.f32 [%rd8],  %f16;
    st.global.f32 [%rd12], %f17;
    mul.lo.u32 %r20, %r12, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    st.global.f32 [%rd8],  %f18;
    st.global.f32 [%rd12], %f19;
    mul.lo.u32 %r20, %r13, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    st.global.f32 [%rd8],  %f20;
    st.global.f32 [%rd12], %f21;
    mul.lo.u32 %r20, %r14, 4;
    cvt.u64.u32 %rd4, %r20;
    add.u64 %rd8,  %rd0, %rd4;
    add.u64 %rd12, %rd1, %rd4;
    st.global.f32 [%rd8],  %f22;
    st.global.f32 [%rd12], %f23;

DONE:
    ret;
}}
"#
    )
}

/// CNOT kernel: swaps amplitudes where ctrl=1,tgt=0 with ctrl=1,tgt=1.
pub fn statevec_apply_cnot_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    format!(
        r#"{header}
// statevec_apply_cnot: CNOT gate — swap amp[ctrl=1,tgt=0] with amp[ctrl=1,tgt=1]
// Parameters: amp_re *f32, amp_im *f32, ctrl_mask u32, tgt_mask u32, n_pairs u32
.visible .entry statevec_apply_cnot(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_ctrl_mask,
    .param .u32 param_tgt_mask,
    .param .u32 param_n_pairs
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<8>;
    .reg .pred %p0, %p1;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n_pairs];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_ctrl_mask];
    ld.param.u32 %r6, [param_tgt_mask];

    // only process indices where ctrl bit = 1
    and.b32 %r7, %r3, %r5;
    setp.eq.u32 %p1, %r7, 0;
    @%p1 bra DONE;

    // only process the tgt=0 member of each pair (avoid the double-swap race:
    // otherwise both the tgt=0 and tgt=1 threads swap the same pair)
    and.b32 %r7, %r3, %r6;
    setp.ne.u32 %p1, %r7, 0;
    @%p1 bra DONE;

    // i0 = index with tgt=0 (clear tgt bit), i1 = index with tgt=1 (set tgt bit)
    not.b32 %r8, %r6;
    and.b32 %r9, %r3, %r8;    // i0 = tid & ~tgt_mask
    or.b32  %r10, %r9, %r6;   // i1 = i0 | tgt_mask

    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    mul.lo.u32 %r11, %r9,  4;
    mul.lo.u32 %r12, %r10, 4;
    cvt.u64.u32 %rd2, %r11;
    cvt.u64.u32 %rd3, %r12;

    add.u64 %rd4, %rd0, %rd2;
    add.u64 %rd5, %rd0, %rd3;
    add.u64 %rd6, %rd1, %rd2;
    add.u64 %rd7, %rd1, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd7];

    st.global.f32 [%rd4], %f2;
    st.global.f32 [%rd6], %f3;
    st.global.f32 [%rd5], %f0;
    st.global.f32 [%rd7], %f1;

DONE:
    ret;
}}
"#
    )
}

/// Pauli-Z expectation value kernel with warp-shuffle reduction.
pub fn expval_pauli_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    format!(
        r#"{header}
// expval_pauli: E = sum_i parity(popcount(i & zmask)) * (re[i]^2 + im[i]^2)
// parity = +1 if popcount even, -1 if odd
// Uses warp-shuffle reduction; result accumulated via atomicAdd to output
// Parameters: amp_re *f32, amp_im *f32, zmask u32, n u32, out *f32
.visible .entry expval_pauli(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_zmask,
    .param .u32 param_n,
    .param .u64 param_out
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_zmask];

    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    mul.lo.u32 %r6, %r3, 4;
    cvt.u64.u32 %rd2, %r6;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd4];

    // prob = re^2 + im^2
    mul.f32 %f2, %f0, %f0;
    mul.f32 %f3, %f1, %f1;
    add.f32 %f4, %f2, %f3;

    // parity via popcount of (i & zmask)
    and.b32 %r7, %r3, %r5;
    popc.b32 %r8, %r7;
    and.b32 %r9, %r8, 1;            // 0 = even parity, 1 = odd parity

    // sign: even -> +prob, odd -> -prob
    neg.f32 %f5, %f4;
    setp.eq.u32 %p0, %r9, 1;
    selp.f32 %f6, %f5, %f4, %p0;

    // warp-shuffle butterfly reduction
    shfl.sync.down.b32 %f7, %f6, 16, 31, 0xffffffff;
    add.f32 %f6, %f6, %f7;
    shfl.sync.down.b32 %f7, %f6, 8, 31, 0xffffffff;
    add.f32 %f6, %f6, %f7;
    shfl.sync.down.b32 %f7, %f6, 4, 31, 0xffffffff;
    add.f32 %f6, %f6, %f7;
    shfl.sync.down.b32 %f7, %f6, 2, 31, 0xffffffff;
    add.f32 %f6, %f6, %f7;
    shfl.sync.down.b32 %f7, %f6, 1, 31, 0xffffffff;
    add.f32 %f6, %f6, %f7;

    // lane 0 writes result
    mov.u32 %r10, %laneid;
    setp.eq.u32 %p0, %r10, 0;
    @!%p0 bra DONE;

    ld.param.u64 %rd5, [param_out];
    atom.global.add.f32 %f7, [%rd5], %f6;

DONE:
    ret;
}}
"#
    )
}

/// Partial trace kernel: accumulates diagonal elements of reduced density matrix.
pub fn partial_trace_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    format!(
        r#"{header}
// partial_trace: each thread computes one reduced-state index, atomically accumulating |amp|^2
// Parameters: amp_re *f32, amp_im *f32, n_total u32, trace_mask u32, n_keep u32, out *f32
.visible .entry partial_trace(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_n_total,
    .param .u32 param_trace_mask,
    .param .u32 param_n_keep,
    .param .u64 param_out
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n_total];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    mul.lo.u32 %r5, %r3, 4;
    cvt.u64.u32 %rd2, %r5;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd4];

    mul.f32 %f2, %f0, %f0;
    mul.f32 %f3, %f1, %f1;
    add.f32 %f4, %f2, %f3;

    // reduced index: compact the kept (non-trace) bits into a dense index.
    ld.param.u32 %r6, [param_trace_mask];
    mov.u32 %r8, 0;          // dense reduced-index accumulator (must start at 0)

    // compact to dense reduced index via parallel bit extract (simplified: use pext-equivalent)
    // For PTX correctness we use the compact mapping via shift:
    // dense_idx = pext(r3, ~trace_mask) — approximate via shift sequence
    // Use bfi/prmt sequence for 2-bit compaction (generalized)
    mov.u32 %r9, 0;
    mov.u32 %r10, 0;
    mov.u32 %r11, 1;
COMPACT_LOOP:
    setp.ge.u32 %p0, %r10, 32;
    @%p0 bra COMPACT_DONE;
    shr.u32 %r12, %r6, %r10;
    and.b32 %r12, %r12, 1;
    setp.eq.u32 %p0, %r12, 1;
    @%p0 bra SKIP_BIT;
    shr.u32 %r12, %r3, %r10;
    and.b32 %r12, %r12, 1;
    shl.b32 %r12, %r12, %r9;
    or.b32  %r8, %r8, %r12;
    add.u32 %r9, %r9, 1;
SKIP_BIT:
    add.u32 %r10, %r10, 1;
    bra COMPACT_LOOP;
COMPACT_DONE:

    mul.lo.u32 %r13, %r8, 4;
    cvt.u64.u32 %rd5, %r13;
    ld.param.u64 %rd6, [param_out];
    add.u64 %rd7, %rd6, %rd5;
    atom.global.add.f32 %f5, [%rd7], %f4;

DONE:
    ret;
}}
"#
    )
}

/// Trotter step kernel: applies ZZ rotation exp(-i*theta*Z⊗Z).
pub fn trotter_step_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let two_pi_hex = format!("0F{:08X}", (2.0_f32 * std::f32::consts::PI).to_bits());
    format!(
        r#"{header}
// trotter_step: apply exp(-i*theta*Z@Z) to amplitude pairs
// parity = popcount(i & zz_mask) & 1; apply phase exp(+/-i*theta)
// Parameters: amp_re *f32, amp_im *f32, zz_mask u32, theta f32, n u32
// 2pi = {two_pi_hex}
.visible .entry trotter_step(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_zz_mask,
    .param .f32 param_theta,
    .param .u32 param_n
)
{{
    .reg .u32 %r<12>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<12>;
    .reg .pred %p0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_zz_mask];
    ld.param.f32 %f0, [param_theta];

    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    mul.lo.u32 %r6, %r3, 4;
    cvt.u64.u32 %rd2, %r6;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.f32 %f1, [%rd3];
    ld.global.f32 %f2, [%rd4];

    // parity of popcount(i & zz_mask)
    and.b32 %r7, %r3, %r5;
    popc.b32 %r8, %r7;
    and.b32 %r9, %r8, 1;

    // odd parity -> phase = exp(+i*theta); even -> exp(-i*theta)
    neg.f32 %f3, %f0;
    setp.eq.u32 %p0, %r9, 1;
    selp.f32 %f4, %f0, %f3, %p0;

    // compute cos(f4) and sin(f4) via rcp approximation (PTX has no direct sin/cos on f32 regs)
    // Use mul.rn + fma chain: cos(x) ~ 1 - x^2/2 + x^4/24, sin(x) ~ x - x^3/6
    mul.f32 %f5, %f4, %f4;             // x^2
    mul.f32 %f6, %f5, %f5;             // x^4
    mov.f32 %f7, 0F3F800000;           // 1.0
    mov.f32 %f8, 0F3F000000;           // 0.5
    mov.f32 %f9, 0F3D2AAAAB;           // 1/24
    // cos ~ 1 - x^2/2 + x^4/24
    mul.f32 %f10, %f5, %f8;
    sub.f32 %f10, %f7, %f10;
    mul.f32 %f11, %f6, %f9;
    add.f32 %f10, %f10, %f11;         // cos(theta)
    // sin ~ x - x^3/6
    mov.f32 %f8, 0F3E2AAAAB;           // 1/6
    mul.f32 %f11, %f5, %f4;            // x^3
    mul.f32 %f11, %f11, %f8;
    sub.f32 %f11, %f4, %f11;           // sin(theta)

    // apply phase: (re + i*im) * (cos + i*sin) = re*cos - im*sin + i*(re*sin + im*cos)
    mul.f32 %f3, %f1, %f10;
    mul.f32 %f4, %f2, %f11;
    sub.f32 %f3, %f3, %f4;             // new re
    mul.f32 %f4, %f1, %f11;
    mul.f32 %f5, %f2, %f10;
    add.f32 %f4, %f4, %f5;             // new im

    st.global.f32 [%rd3], %f3;
    st.global.f32 [%rd4], %f4;

DONE:
    ret;
}}
"#
    )
}

/// Measurement probability kernel: P(qubit=k) via masked select + warp reduce.
pub fn measure_prob_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    format!(
        r#"{header}
// measure_prob: P(qubit=outcome) = sum_{{i: i[qubit]=outcome}} |amp_i|^2
// Parameters: amp_re *f32, amp_im *f32, qubit_mask u32, outcome u32, n u32, out *f32
.visible .entry measure_prob(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_qubit_mask,
    .param .u32 param_outcome,
    .param .u32 param_n,
    .param .u64 param_out
)
{{
    .reg .u32 %r<12>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p0, %p1;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_qubit_mask];
    ld.param.u32 %r6, [param_outcome];

    // Every lane loads its amplitude and computes prob = |amp|^2, then masks the
    // contribution to 0 when this index does NOT have the target qubit value.
    // (We must NOT branch out non-matching lanes before the warp shuffle: that
    // would diverge the warp while the shfl uses a full-warp membermask, which is
    // UB. Instead, zero the contribution and let all lanes reduce.)
    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    mul.lo.u32 %r8, %r3, 4;
    cvt.u64.u32 %rd2, %r8;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;

    ld.global.f32 %f0, [%rd3];
    ld.global.f32 %f1, [%rd4];

    mul.f32 %f2, %f0, %f0;
    mul.f32 %f3, %f1, %f1;
    add.f32 %f4, %f2, %f3;

    // measured bit value (0/1) and match against outcome
    and.b32 %r7, %r3, %r5;
    setp.ne.u32 %p1, %r7, 0;       // p1 = (bit == 1)
    selp.u32 %r9, 1, 0, %p1;       // r9 = measured bit value
    setp.eq.u32 %p1, %r9, %r6;     // p1 = (bit == outcome) -> matches
    mov.f32 %f6, 0F00000000;
    selp.f32 %f4, %f4, %f6, %p1;   // contribution = match ? prob : 0

    // warp-shuffle reduction
    shfl.sync.down.b32 %f5, %f4, 16, 31, 0xffffffff;
    add.f32 %f4, %f4, %f5;
    shfl.sync.down.b32 %f5, %f4, 8, 31, 0xffffffff;
    add.f32 %f4, %f4, %f5;
    shfl.sync.down.b32 %f5, %f4, 4, 31, 0xffffffff;
    add.f32 %f4, %f4, %f5;
    shfl.sync.down.b32 %f5, %f4, 2, 31, 0xffffffff;
    add.f32 %f4, %f4, %f5;
    shfl.sync.down.b32 %f5, %f4, 1, 31, 0xffffffff;
    add.f32 %f4, %f4, %f5;

    mov.u32 %r9, %laneid;
    setp.eq.u32 %p0, %r9, 0;
    @!%p0 bra DONE;

    ld.param.u64 %rd5, [param_out];
    atom.global.add.f32 %f5, [%rd5], %f4;

DONE:
    ret;
}}
"#
    )
}

/// Radix-2 Fourier butterfly kernel with a phase twiddle (one QFT stage).
///
/// For each amplitude pair `(x0, x1)` selected by `qubit` (mask), computes the
/// twiddled Cooley-Tukey butterfly
///
/// ```text
/// w  = e^{iθ} = cos θ + i sin θ
/// y0 = (x0 + w·x1) / √2
/// y1 = (x0 − w·x1) / √2
/// ```
///
/// The pair-index reconstruction prologue mirrors `statevec_apply_1q`, and the
/// complex multiply `w·x1` reuses the cos/sin polynomial-twiddle pattern from
/// `trotter_step`.
pub fn qft_butterfly_ptx(sm: u32) -> String {
    let header = ptx_header(sm);
    let inv_sqrt2_hex = format!("0F{:08X}", (std::f32::consts::FRAC_1_SQRT_2).to_bits());
    format!(
        r#"{header}
// qft_butterfly: y0 = (x0 + w*x1)/sqrt(2), y1 = (x0 - w*x1)/sqrt(2), w = exp(i*theta)
// Parameters: amp_re *f32, amp_im *f32, mask u32, theta f32, n_pairs u32
// inv_sqrt2 = {inv_sqrt2_hex}
.visible .entry qft_butterfly(
    .param .u64 param_amp_re,
    .param .u64 param_amp_im,
    .param .u32 param_mask,
    .param .f32 param_theta,
    .param .u32 param_n_pairs
)
{{
    .reg .u32 %r<16>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<32>;
    .reg .pred %p0;

    // tid = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    // stride loop guard
    ld.param.u32 %r4, [param_n_pairs];
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra DONE;

    ld.param.u32 %r5, [param_mask];

    // i0 = ((tid & ~(mask-1)) << 1) | (tid & (mask-1))  (insert a 0 bit at the qubit slot)
    sub.u32 %r8, %r5, 1;       // mask - 1 (low-bit mask)
    not.b32 %r6, %r8;          // ~(mask-1) (high-bit mask)
    and.b32 %r7, %r3, %r6;
    shl.b32 %r7, %r7, 1;
    and.b32 %r9, %r3, %r8;
    or.b32 %r10, %r7, %r9;

    // i1 = i0 | mask
    or.b32 %r11, %r10, %r5;

    // load base pointers
    ld.param.u64 %rd0, [param_amp_re];
    ld.param.u64 %rd1, [param_amp_im];

    // byte offsets: i0*4, i1*4
    mul.lo.u32 %r12, %r10, 4;
    mul.lo.u32 %r13, %r11, 4;
    cvt.u64.u32 %rd2, %r12;
    cvt.u64.u32 %rd3, %r13;

    add.u64 %rd4, %rd0, %rd2;   // &re[i0]
    add.u64 %rd5, %rd0, %rd3;   // &re[i1]
    add.u64 %rd6, %rd1, %rd2;   // &im[i0]
    add.u64 %rd7, %rd1, %rd3;   // &im[i1]

    // x0 = amp[i0], x1 = amp[i1]
    ld.global.f32 %f0, [%rd4];   // x0.re
    ld.global.f32 %f1, [%rd6];   // x0.im
    ld.global.f32 %f2, [%rd5];   // x1.re
    ld.global.f32 %f3, [%rd7];   // x1.im

    // twiddle w = cos(theta) + i sin(theta) via polynomial approximation
    ld.param.f32 %f4, [param_theta];
    mul.f32 %f5, %f4, %f4;             // x^2
    mul.f32 %f6, %f5, %f5;             // x^4
    mov.f32 %f7, 0F3F800000;           // 1.0
    mov.f32 %f8, 0F3F000000;           // 0.5
    mov.f32 %f9, 0F3D2AAAAB;           // 1/24
    mul.f32 %f10, %f5, %f8;
    sub.f32 %f10, %f7, %f10;
    mul.f32 %f11, %f6, %f9;
    add.f32 %f10, %f10, %f11;          // cos(theta)
    mov.f32 %f8, 0F3E2AAAAB;           // 1/6
    mul.f32 %f11, %f5, %f4;            // x^3
    mul.f32 %f11, %f11, %f8;
    sub.f32 %f11, %f4, %f11;           // sin(theta)

    // wx = w * x1 = (cos + i sin)(x1.re + i x1.im)
    // re(wx) = cos*x1re - sin*x1im ; im(wx) = cos*x1im + sin*x1re
    mul.f32 %f12, %f10, %f2;
    mul.f32 %f13, %f11, %f3;
    sub.f32 %f14, %f12, %f13;          // re(wx)
    mul.f32 %f12, %f10, %f3;
    mul.f32 %f13, %f11, %f2;
    add.f32 %f15, %f12, %f13;          // im(wx)

    // load 1/sqrt(2)
    mov.f32 %f16, {inv_sqrt2_hex};

    // y0 = (x0 + wx) / sqrt(2)
    add.f32 %f17, %f0, %f14;
    mul.f32 %f17, %f17, %f16;          // y0.re
    add.f32 %f18, %f1, %f15;
    mul.f32 %f18, %f18, %f16;          // y0.im

    // y1 = (x0 - wx) / sqrt(2)
    sub.f32 %f19, %f0, %f14;
    mul.f32 %f19, %f19, %f16;          // y1.re
    sub.f32 %f20, %f1, %f15;
    mul.f32 %f20, %f20, %f16;          // y1.im

    // store results
    st.global.f32 [%rd4], %f17;
    st.global.f32 [%rd6], %f18;
    st.global.f32 [%rd5], %f19;
    st.global.f32 [%rd7], %f20;

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
    fn ptx_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!statevec_apply_1q_ptx(sm).is_empty(), "sm={sm}");
            assert!(!statevec_apply_2q_ptx(sm).is_empty(), "sm={sm}");
            assert!(!statevec_apply_cnot_ptx(sm).is_empty(), "sm={sm}");
            assert!(!expval_pauli_ptx(sm).is_empty(), "sm={sm}");
            assert!(!partial_trace_ptx(sm).is_empty(), "sm={sm}");
            assert!(!trotter_step_ptx(sm).is_empty(), "sm={sm}");
            assert!(!measure_prob_ptx(sm).is_empty(), "sm={sm}");
            assert!(!qft_butterfly_ptx(sm).is_empty(), "sm={sm}");
        }
    }

    #[test]
    fn ptx_contains_target_string() {
        let ptx = statevec_apply_1q_ptx(80);
        assert!(ptx.contains("sm_80"), "missing sm_80 in PTX");
        assert!(ptx.contains(".version 8.0"), "missing version in PTX");
    }
}
