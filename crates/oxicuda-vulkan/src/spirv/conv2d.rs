//! SPIR-V compute-shader generator for 2-D convolution (NCHW layout).

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, OP_F_ADD, OP_F_MUL, OP_I_ADD, OP_I_MUL, OP_I_SUB, OP_LOGICAL_AND,
    OP_U_DIV, OP_U_LESS_THAN, OP_U_MOD, SPIRV_VERSION_1_3, STORAGE_CLASS_FUNCTION,
};
use super::preamble::{emit_float_ssbo, emit_preamble, load_gid_x};

/// Generate a SPIR-V compute shader for 2-D convolution (NCHW layout).
///
/// One invocation per output element.  Triple-nested accumulation loop over
/// `(in_channel, filter_y, filter_x)` with unsigned-arithmetic padding checks.
///
/// Bindings: 0 = input `float[]`, 1 = filter `float[]`, 2 = output `float[]`.
/// All dimension constants are baked into the SPIR-V binary.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_spirv(
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    k_out: u32,
    fh: u32,
    fw: u32,
    oh: u32,
    ow: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, filter_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 2);

    // Baked constants
    let c_cin = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_cin, c_in);
    let c_hin = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_hin, h_in);
    let c_win = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_win, w_in);
    let c_kout = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_kout, k_out);
    let c_fh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fh, fh);
    let c_fw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_fw, fw);
    let c_oh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_oh, oh);
    let c_ow = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_ow, ow);
    let c_sh = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sh, stride_h);
    let c_sw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sw, stride_w);
    let c_ph = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_ph, pad_h);
    let c_pw = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_pw, pad_w);
    let total = n
        .saturating_mul(k_out)
        .saturating_mul(oh)
        .saturating_mul(ow);
    let c_total = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_total, total);

    // Labels
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_merge = m.alloc_id();
    let lbl_ci_h = m.alloc_id();
    let lbl_ci_b = m.alloc_id();
    let lbl_ci_c = m.alloc_id();
    let lbl_ci_m = m.alloc_id();
    let lbl_fy_h = m.alloc_id();
    let lbl_fy_b = m.alloc_id();
    let lbl_fy_c = m.alloc_id();
    let lbl_fy_m = m.alloc_id();
    let lbl_fx_h = m.alloc_id();
    let lbl_fx_b = m.alloc_id();
    let lbl_fx_c = m.alloc_id();
    let lbl_fx_m = m.alloc_id();
    let lbl_ib = m.alloc_id();
    let lbl_ib_m = m.alloc_id();

    // ── Function ──
    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(lbl_entry);

    // SPIR-V requires all Function-storage OpVariables to be declared in the
    // first (entry) basic block, so allocate them here before any branch.
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    let var_ci = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_ci, STORAGE_CLASS_FUNCTION);
    let var_fy = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_fy, STORAGE_CLASS_FUNCTION);
    let var_fx = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_fx, STORAGE_CLASS_FUNCTION);

    let gid = load_gid_x(&mut m, &b);
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, c_total]);
    m.emit_selection_merge(lbl_merge);
    m.emit_branch_conditional(cond, lbl_body, lbl_merge);

    m.emit_label(lbl_body);

    // Decompose gid → (batch, kf, oy, ox)
    let ox_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, ox_val, gid, c_ow]);
    let t1 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, t1, gid, c_ow]);
    let oy_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, oy_val, t1, c_oh]);
    let t2 = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, t2, t1, c_oh]);
    let kf = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, kf, t2, c_kout]);
    let batch = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, batch, t2, c_kout]);

    // Initialise the accumulator and channel counter (variables declared in the
    // entry block); the fy/fx counters are initialised inside their loops.
    m.emit_store(var_acc, b.c_float_0);
    m.emit_store(var_ci, b.c_uint_0);

    m.emit_branch(lbl_ci_h);

    // ── ci loop ──
    m.emit_label(lbl_ci_h);
    let ci = m.alloc_id();
    m.emit_load(b.ty_uint, ci, var_ci);
    let ci_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ci_ok, ci, c_cin]);
    m.emit_loop_merge(lbl_ci_m, lbl_ci_c);
    m.emit_branch_conditional(ci_ok, lbl_ci_b, lbl_ci_m);

    m.emit_label(lbl_ci_b);
    m.emit_store(var_fy, b.c_uint_0);
    m.emit_branch(lbl_fy_h);

    // ── fy loop ──
    m.emit_label(lbl_fy_h);
    let fy = m.alloc_id();
    m.emit_load(b.ty_uint, fy, var_fy);
    let fy_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, fy_ok, fy, c_fh]);
    m.emit_loop_merge(lbl_fy_m, lbl_fy_c);
    m.emit_branch_conditional(fy_ok, lbl_fy_b, lbl_fy_m);

    m.emit_label(lbl_fy_b);
    m.emit_store(var_fx, b.c_uint_0);
    m.emit_branch(lbl_fx_h);

    // ── fx loop ──
    m.emit_label(lbl_fx_h);
    let fx = m.alloc_id();
    m.emit_load(b.ty_uint, fx, var_fx);
    let fx_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, fx_ok, fx, c_fw]);
    m.emit_loop_merge(lbl_fx_m, lbl_fx_c);
    m.emit_branch_conditional(fx_ok, lbl_fx_b, lbl_fx_m);

    m.emit_label(lbl_fx_b);

    // iy = oy*stride_h + fy − pad_h (unsigned wrapping)
    let oy_sh = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, oy_sh, oy_val, c_sh]);
    let oy_sh_fy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oy_sh_fy, oy_sh, fy]);
    let iy = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, iy, oy_sh_fy, c_ph]);

    // ix = ox*stride_w + fx − pad_w
    let ox_sw = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, ox_sw, ox_val, c_sw]);
    let ox_sw_fx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ox_sw_fx, ox_sw, fx]);
    let ix = m.alloc_id();
    m.emit(OP_I_SUB, &[b.ty_uint, ix, ox_sw_fx, c_pw]);

    // Bounds: iy < h_in && ix < w_in (unsigned catches underflow)
    let iy_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, iy_ok, iy, c_hin]);
    let ix_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, ix_ok, ix, c_win]);
    let ok = m.alloc_id();
    m.emit(OP_LOGICAL_AND, &[b.ty_bool, ok, iy_ok, ix_ok]);

    m.emit_selection_merge(lbl_ib_m);
    m.emit_branch_conditional(ok, lbl_ib, lbl_ib_m);

    m.emit_label(lbl_ib);

    // input_idx = ((batch*c_in + ci)*h_in + iy)*w_in + ix
    let bc = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bc, batch, c_cin]);
    let bc_ci = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, bc_ci, bc, ci]);
    let bch = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bch, bc_ci, c_hin]);
    let bch_iy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, bch_iy, bch, iy]);
    let bchw = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bchw, bch_iy, c_win]);
    let in_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, in_idx, bchw, ix]);

    // filter_idx = ((kf*c_in + ci)*fh + fy)*fw + fx
    let kc = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kc, kf, c_cin]);
    let kc_ci = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, kc_ci, kc, ci]);
    let kcf = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kcf, kc_ci, c_fh]);
    let kcf_fy = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, kcf_fy, kcf, fy]);
    let kcff = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kcff, kcf_fy, c_fw]);
    let f_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, f_idx, kcff, fx]);

    // Load input and filter, accumulate
    let inp_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, inp_ptr, input_var, &[b.c_uint_0, in_idx]);
    let inp_v = m.alloc_id();
    m.emit_load(b.ty_float, inp_v, inp_ptr);
    let flt_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, flt_ptr, filter_var, &[b.c_uint_0, f_idx]);
    let flt_v = m.alloc_id();
    m.emit_load(b.ty_float, flt_v, flt_ptr);
    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, inp_v, flt_v]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(lbl_ib_m);
    m.emit_label(lbl_ib_m);

    // fx continue / merge
    m.emit_branch(lbl_fx_c);
    m.emit_label(lbl_fx_c);
    let fx_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, fx_inc, fx, b.c_uint_1]);
    m.emit_store(var_fx, fx_inc);
    m.emit_branch(lbl_fx_h);
    m.emit_label(lbl_fx_m);

    // fy continue / merge
    m.emit_branch(lbl_fy_c);
    m.emit_label(lbl_fy_c);
    let fy_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, fy_inc, fy, b.c_uint_1]);
    m.emit_store(var_fy, fy_inc);
    m.emit_branch(lbl_fy_h);
    m.emit_label(lbl_fy_m);

    // ci continue / merge
    m.emit_branch(lbl_ci_c);
    m.emit_label(lbl_ci_c);
    let ci_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ci_inc, ci, b.c_uint_1]);
    m.emit_store(var_ci, ci_inc);
    m.emit_branch(lbl_ci_h);

    m.emit_label(lbl_ci_m);

    // Store result
    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, final_acc);

    m.emit_branch(lbl_merge);
    m.emit_label(lbl_merge);
    m.emit_return();
    m.emit_function_end();
    m.finalize()
}
