//! Shared kernel body for standard (multi-channel, `groups >= 1`) forward
//! convolution.
//!
//! Both the 1x1 ([`Conv1x1`](super::direct::Conv1x1)) and the general
//! implicit-GEMM ([`ImplicitGemmConv`](super::implicit_gemm::ImplicitGemmConv))
//! forward engines share the exact same per-output-element compute: one thread
//! owns one output pixel `(n, k, oh, ow)` and accumulates the cross-correlation
//!
//! ```text
//! out[n, k, oh, ow] = Σ_{cg} Σ_r Σ_s in[n, c_in, ih, iw] * filter[k, cg, r, s]
//! c_in = group * (C_in / groups) + cg
//! ih   = oh*stride_h - pad_h + r*dilation_h
//! iw   = ow*stride_w - pad_w + s*dilation_w
//! group = k / (C_out / groups)
//! ```
//!
//! over the in-bounds taps (out-of-range taps contribute zero — implicit zero
//! padding). This is the **cross-correlation** convention used by cuDNN (no
//! 180° kernel flip), matching the depthwise reference in
//! [`super::direct`].
//!
//! The reduction over the input channels of the group is a real PTX runtime
//! loop (the channel count is data, not a code-gen constant we want to unroll
//! for large `C`), while the filter spatial window `R × S` is unrolled at
//! code-gen time. All tensor geometry (dims, padding, stride, dilation) is read
//! from kernel parameters; the filter extent, channels-per-group and layout are
//! known when the PTX is generated.
//!
//! Two memory layouts are supported, selected at code-gen time from the
//! [`TensorLayout`](crate::types::TensorLayout):
//!
//! * **NCHW** — activations `[N, C, H, W]`, filter `[K, C/g, R, S]` (row-major).
//! * **NHWC** — activations `[N, H, W, C]`, filter `[K, R, S, C/g]` (the layout
//!   produced by [`TensorDesc::nhwc`](crate::types::TensorDesc::nhwc), whose
//!   strides place the channel last).

use oxicuda_ptx::builder::{BodyBuilder, KernelBuilder};
use oxicuda_ptx::ir::{PtxType, Register};

/// Adds the full parameter set consumed by [`emit_standard_conv_body`] to a
/// [`KernelBuilder`].
///
/// Keeping the list in one place guarantees the two engines that share the
/// body (`Conv1x1` and `ImplicitGemmConv`) declare exactly the parameters the
/// emitter loads, and that the host launch argument order matches.
#[must_use]
pub(crate) fn with_standard_conv_params(kb: KernelBuilder) -> KernelBuilder {
    kb.param("input", PtxType::U64)
        .param("filter", PtxType::U64)
        .param("output", PtxType::U64)
        .param("bias", PtxType::U64)
        .param("in_channels", PtxType::U32)
        .param("out_channels", PtxType::U32)
        .param("in_h", PtxType::U32)
        .param("in_w", PtxType::U32)
        .param("out_h", PtxType::U32)
        .param("out_w", PtxType::U32)
        .param("pad_h", PtxType::U32)
        .param("pad_w", PtxType::U32)
        .param("stride_h", PtxType::U32)
        .param("stride_w", PtxType::U32)
        .param("dilation_h", PtxType::U32)
        .param("dilation_w", PtxType::U32)
        .param("total_outputs", PtxType::U32)
}

/// Emits the standard-convolution kernel body.
///
/// `channels_last` selects NHWC (`true`) vs NCHW (`false`) addressing.
/// `in_ch_per_group` / `out_ch_per_group` are `C_in / groups` and
/// `C_out / groups`; `filter_h` / `filter_w` are the filter spatial extent.
pub(crate) fn emit_standard_conv_body(
    b: &mut BodyBuilder<'_>,
    elem_ty: PtxType,
    channels_last: bool,
    filter_h: u32,
    filter_w: u32,
    in_ch_per_group: u32,
    out_ch_per_group: u32,
) {
    b.comment("=== Standard multi-channel convolution (cross-correlation) ===");
    b.comment("One thread per output element; reduce over (C_in/groups x R x S).");

    let gid = b.global_thread_id_x();
    let total = b.load_param_u32("total_outputs");
    let gid_cmp = gid.clone();
    b.if_lt_u32(gid_cmp, total, move |b| {
        emit_standard_conv_pixel(
            b,
            elem_ty,
            channels_last,
            filter_h,
            filter_w,
            in_ch_per_group,
            out_ch_per_group,
            gid,
        );
    });

    b.ret();
}

/// Emits the body executed by a single in-range output thread.
#[allow(clippy::too_many_arguments)]
fn emit_standard_conv_pixel(
    b: &mut BodyBuilder<'_>,
    elem_ty: PtxType,
    channels_last: bool,
    filter_h: u32,
    filter_w: u32,
    in_ch_per_group: u32,
    out_ch_per_group: u32,
    gid: Register,
) {
    let elem_bytes = elem_ty.size_bytes() as u32;

    // Tensor base pointers.
    let input_ptr = b.load_param_u64("input");
    let filter_ptr = b.load_param_u64("filter");
    let output_ptr = b.load_param_u64("output");
    let bias_ptr = b.load_param_u64("bias");

    // Geometry parameters.
    let in_channels = b.load_param_u32("in_channels");
    let out_channels = b.load_param_u32("out_channels");
    let in_h = b.load_param_u32("in_h");
    let in_w = b.load_param_u32("in_w");
    let out_h = b.load_param_u32("out_h");
    let out_w = b.load_param_u32("out_w");
    let pad_h = b.load_param_u32("pad_h");
    let pad_w = b.load_param_u32("pad_w");
    let stride_h = b.load_param_u32("stride_h");
    let stride_w = b.load_param_u32("stride_w");
    let dil_h = b.load_param_u32("dilation_h");
    let dil_w = b.load_param_u32("dilation_w");

    // Decompose the linear output index (enumerated in N,K,P,Q nesting order,
    // i.e. ow fastest) into (n, k, oh, ow). For NCHW this linear index *is* the
    // output memory offset; for NHWC the memory offset is recomputed below.
    b.comment("Decompose linear output index -> (n, k, oh, ow)");
    let ow = b.alloc_reg(PtxType::U32);
    let t1 = b.alloc_reg(PtxType::U32);
    let oh = b.alloc_reg(PtxType::U32);
    let t2 = b.alloc_reg(PtxType::U32);
    let k = b.alloc_reg(PtxType::U32);
    let n = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("rem.u32 {ow}, {gid}, {out_w};"));
    b.raw_ptx(&format!("div.u32 {t1}, {gid}, {out_w};"));
    b.raw_ptx(&format!("rem.u32 {oh}, {t1}, {out_h};"));
    b.raw_ptx(&format!("div.u32 {t2}, {t1}, {out_h};"));
    b.raw_ptx(&format!("rem.u32 {k}, {t2}, {out_channels};"));
    b.raw_ptx(&format!("div.u32 {n}, {t2}, {out_channels};"));

    // Group routing: group = k / out_ch_per_group, cg_start = group * icpg.
    let icpg_reg = b.mov_imm_u32(in_ch_per_group);
    let ocpg_reg = b.mov_imm_u32(out_ch_per_group);
    let group = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("div.u32 {group}, {k}, {ocpg_reg};"));
    let cg_start = b.mul_lo_u32(group, icpg_reg.clone());

    // Layout-dependent per-thread invariants:
    //   NCHW filter [K, icpg, R, S]: row block base = k * icpg.
    //   NHWC filter [K, R, S, icpg]: k * R (the outer filter-row factor).
    let rxs = filter_h.saturating_mul(filter_w);
    let rxs_reg = b.mov_imm_u32(rxs);
    let r_extent_reg = b.mov_imm_u32(filter_h);
    let s_extent_reg = b.mov_imm_u32(filter_w);
    let (k_icpg, k_r) = if channels_last {
        (None, Some(b.mul_lo_u32(k.clone(), r_extent_reg.clone())))
    } else {
        (Some(b.mul_lo_u32(k.clone(), icpg_reg.clone())), None)
    };

    // Accumulator and a reusable zero of the kernel's float width. The
    // accumulator is read-modify-written across the runtime channel loop, so it
    // must be a single fixed register (never the SSA-style fma helper, which
    // allocates a fresh destination each call).
    let acc = b.alloc_reg(elem_ty);
    let fzero = b.alloc_reg(elem_ty);
    let (zero_lit, fma_op, add_op) = float_ops(elem_ty);
    b.raw_ptx(&format!(
        "mov.{0} {acc}, {zero_lit};",
        elem_ty.as_ptx_str().trim_start_matches('.')
    ));
    b.raw_ptx(&format!(
        "mov.{0} {fzero}, {zero_lit};",
        elem_ty.as_ptx_str().trim_start_matches('.')
    ));

    // Runtime reduction over the input channels of this group.
    b.comment("Reduce over input channels of the group (runtime loop)");
    let cg = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {cg}, 0;"));
    let loop_start = b.fresh_label("conv_cg");
    let loop_end = b.fresh_label("conv_cg_end");
    b.raw_ptx(&format!("{loop_start}:"));
    let p_done = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.hs.u32 {p_done}, {cg}, {icpg_reg};"));
    b.raw_ptx(&format!("@{p_done} bra {loop_end};"));

    // c_in = cg_start + cg.
    let c_in = b.add_u32(cg_start.clone(), cg.clone());

    // Per-channel addressing invariants.
    //   NCHW input row factor: nc = n * C_in + c_in.
    //   NCHW filter base:       fbase = (k*icpg + cg) * (R*S).
    let nc = if channels_last {
        None
    } else {
        Some(b.mad_lo_u32(n.clone(), in_channels.clone(), c_in.clone()))
    };
    let f_base = match &k_icpg {
        Some(k_icpg_reg) => {
            let frow = b.add_u32(k_icpg_reg.clone(), cg.clone());
            Some(b.mul_lo_u32(frow, rxs_reg.clone()))
        }
        None => None,
    };

    b.comment("Unrolled cross-correlation over the R x S filter window");
    for r in 0..filter_h {
        // ih = oh*stride_h - pad_h + r*dilation_h  (signed).
        let r_reg = b.mov_imm_u32(r);
        let rdil = b.mul_lo_u32(dil_h.clone(), r_reg);
        let ih_pos = b.mad_lo_u32(oh.clone(), stride_h.clone(), rdil);
        let ih = b.alloc_reg(PtxType::S32);
        b.raw_ptx(&format!("sub.s32 {ih}, {ih_pos}, {pad_h};"));
        // Unsigned compare doubles as a 0 <= ih < H bounds check.
        let p_ih = b.alloc_reg(PtxType::Pred);
        b.raw_ptx(&format!("setp.lo.u32 {p_ih}, {ih}, {in_h};"));

        for s in 0..filter_w {
            let s_reg = b.mov_imm_u32(s);
            let sdil = b.mul_lo_u32(dil_w.clone(), s_reg);
            let iw_pos = b.mad_lo_u32(ow.clone(), stride_w.clone(), sdil);
            let iw = b.alloc_reg(PtxType::S32);
            b.raw_ptx(&format!("sub.s32 {iw}, {iw_pos}, {pad_w};"));
            let p_iw = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.lo.u32 {p_iw}, {iw}, {in_w};"));
            let p_valid = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("and.pred {p_valid}, {p_ih}, {p_iw};"));

            // Input linear index (layout-dependent).
            let in_idx = if channels_last {
                // ((n*H + ih)*W + iw)*C + c_in
                let row = b.mad_lo_u32(n.clone(), in_h.clone(), ih.clone());
                let sp = b.mad_lo_u32(row, in_w.clone(), iw.clone());
                b.mad_lo_u32(sp, in_channels.clone(), c_in.clone())
            } else {
                // ((n*C + c_in)*H + ih)*W + iw
                let nc_reg = nc.clone().unwrap_or_else(|| c_in.clone());
                let row = b.mad_lo_u32(nc_reg, in_h.clone(), ih.clone());
                b.mad_lo_u32(row, in_w.clone(), iw.clone())
            };
            // Clamp to 0 when out of range so the load address is always valid;
            // the loaded value is masked to zero afterwards.
            let zero_idx = b.mov_imm_u32(0);
            let safe_idx = b.selp(PtxType::U32, in_idx, zero_idx, p_valid.clone());
            let in_addr = b.byte_offset_addr(input_ptr.clone(), safe_idx, elem_bytes);
            let raw_val = load_float(b, elem_ty, in_addr);
            let val = b.selp(elem_ty, raw_val, fzero.clone(), p_valid);

            // Filter linear index (layout-dependent), always in range.
            let filt_idx = if channels_last {
                // ((k*R + r)*S + s)*icpg + cg
                let kr_reg = k_r.clone().unwrap_or_else(|| k.clone());
                let r_off = b.mov_imm_u32(r);
                let kr = b.add_u32(kr_reg, r_off);
                let s_off = b.mov_imm_u32(s);
                let krs = b.mad_lo_u32(kr, s_extent_reg.clone(), s_off);
                b.mad_lo_u32(krs, icpg_reg.clone(), cg.clone())
            } else {
                // (k*icpg + cg) * (R*S) + (r*S + s)
                let f_base_reg = f_base.clone().unwrap_or_else(|| cg.clone());
                let f_off = r.saturating_mul(filter_w).saturating_add(s);
                let f_off_reg = b.mov_imm_u32(f_off);
                b.add_u32(f_base_reg, f_off_reg)
            };
            let filt_addr = b.byte_offset_addr(filter_ptr.clone(), filt_idx, elem_bytes);
            let weight = load_float(b, elem_ty, filt_addr);

            // acc += val * weight  (read-modify-write the fixed accumulator).
            b.raw_ptx(&format!("{fma_op} {acc}, {val}, {weight}, {acc};"));
        }
    }

    // cg += 1; loop.
    b.raw_ptx(&format!("add.u32 {cg}, {cg}, 1;"));
    b.raw_ptx(&format!("bra {loop_start};"));
    b.raw_ptx(&format!("{loop_end}:"));

    // Optional per-output-channel bias: out += bias[k] when the pointer is
    // non-null. The host passes 0 when no bias is supplied.
    b.comment("Guarded per-output-channel bias add");
    let no_bias = b.fresh_label("conv_no_bias");
    let p_has_bias = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.ne.u64 {p_has_bias}, {bias_ptr}, 0;"));
    b.raw_ptx(&format!("@!{p_has_bias} bra {no_bias};"));
    let bias_addr = b.byte_offset_addr(bias_ptr, k.clone(), elem_bytes);
    let bias_val = load_float(b, elem_ty, bias_addr);
    b.raw_ptx(&format!("{add_op} {acc}, {acc}, {bias_val};"));
    b.raw_ptx(&format!("{no_bias}:"));

    // Output linear index (layout-dependent) and store.
    let out_idx = if channels_last {
        // ((n*P + oh)*Q + ow)*K + k
        let row = b.mad_lo_u32(n, out_h, oh);
        let sp = b.mad_lo_u32(row, out_w, ow);
        b.mad_lo_u32(sp, out_channels, k)
    } else {
        // NCHW: the enumeration order equals the memory offset.
        gid
    };
    let out_addr = b.byte_offset_addr(output_ptr, out_idx, elem_bytes);
    store_float(b, elem_ty, out_addr, acc);
}

/// Returns the `(zero-literal, fma op, add op)` PTX mnemonics for the float
/// width. Both the depthwise reference and the standard kernel round-to-nearest.
fn float_ops(elem_ty: PtxType) -> (&'static str, &'static str, &'static str) {
    if elem_ty == PtxType::F64 {
        ("0d0000000000000000", "fma.rn.f64", "add.rn.f64")
    } else {
        ("0f00000000", "fma.rn.f32", "add.rn.f32")
    }
}

/// Loads one scalar of the kernel's float width from global memory.
fn load_float(b: &mut BodyBuilder<'_>, elem_ty: PtxType, addr: Register) -> Register {
    if elem_ty == PtxType::F64 {
        b.load_global_f64(addr)
    } else {
        b.load_global_f32(addr)
    }
}

/// Stores one scalar of the kernel's float width to global memory.
fn store_float(b: &mut BodyBuilder<'_>, elem_ty: PtxType, addr: Register, val: Register) {
    if elem_ty == PtxType::F64 {
        b.store_global_f64(addr, val);
    } else {
        b.store_global_f32(addr, val);
    }
}
