//! SPIR-V compute-shader generator for scaled dot-product attention.

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, GLSL_EXP, GLSL_F_MAX, OP_F_ADD, OP_F_DIV, OP_F_MUL, OP_F_SUB, OP_I_ADD,
    OP_I_MUL, OP_U_DIV, OP_U_LESS_THAN, OP_U_MOD, SPIRV_VERSION_1_3, STORAGE_CLASS_FUNCTION,
};
use super::preamble::{emit_float_ssbo, emit_preamble, load_gid_x};

/// Generate a SPIR-V compute shader for scaled dot-product attention.
///
/// Each invocation handles one `(batch_head, query_position)`.  The shader
/// performs numerically-stable softmax with optional causal masking.
///
/// Bindings: 0 = Q `float[]`, 1 = K `float[]`, 2 = V `float[]`,
/// 3 = O `float[]`.  All dimension constants are baked in.
pub fn attention_spirv(
    batch_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    head_dim: u32,
    scale: f32,
    causal: bool,
) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, q_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, k_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, v_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, o_var) = emit_float_ssbo(&mut m, &b, 3);

    // Baked constants
    let c_sq = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_sq, seq_q);
    let c_skv = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_skv, seq_kv);
    let c_hd = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_hd, head_dim);
    let c_scale = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_scale, scale);
    let c_neg_inf = m.alloc_id();
    m.emit_constant_f32(b.ty_float, c_neg_inf, f32::NEG_INFINITY);
    let c_total = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_total, batch_heads.saturating_mul(seq_q));

    // ── Labels ──
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_merge = m.alloc_id();
    // Pass 1 labels (max-score)
    let lbl_s1h = m.alloc_id();
    let lbl_s1b = m.alloc_id();
    let lbl_s1c = m.alloc_id();
    let lbl_s1m = m.alloc_id();
    let lbl_s1w = m.alloc_id();
    let lbl_s1wm = m.alloc_id();
    let lbl_d1h = m.alloc_id();
    let lbl_d1b = m.alloc_id();
    let lbl_d1c = m.alloc_id();
    let lbl_d1m = m.alloc_id();
    // Zero-output labels
    let lbl_zh = m.alloc_id();
    let lbl_zb = m.alloc_id();
    let lbl_zc = m.alloc_id();
    let lbl_zm = m.alloc_id();
    // Pass 2 labels (accumulate)
    let lbl_s2h = m.alloc_id();
    let lbl_s2b = m.alloc_id();
    let lbl_s2c = m.alloc_id();
    let lbl_s2m = m.alloc_id();
    let lbl_s2w = m.alloc_id();
    let lbl_s2wm = m.alloc_id();
    let lbl_d2h = m.alloc_id();
    let lbl_d2b = m.alloc_id();
    let lbl_d2c = m.alloc_id();
    let lbl_d2m = m.alloc_id();
    let lbl_d3h = m.alloc_id();
    let lbl_d3b = m.alloc_id();
    let lbl_d3c = m.alloc_id();
    let lbl_d3m = m.alloc_id();
    // Normalize labels
    let lbl_d4h = m.alloc_id();
    let lbl_d4b = m.alloc_id();
    let lbl_d4c = m.alloc_id();
    let lbl_d4m = m.alloc_id();

    // ── Function ──
    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(lbl_entry);

    // SPIR-V requires all Function-storage OpVariables to be declared in the
    // first (entry) basic block, so allocate them here before any branch.
    let var_max = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_max, STORAGE_CLASS_FUNCTION);
    let var_sum = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_sum, STORAGE_CLASS_FUNCTION);
    let var_dot = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_dot, STORAGE_CLASS_FUNCTION);
    let var_sk = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_sk, STORAGE_CLASS_FUNCTION);
    let var_d = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_d, STORAGE_CLASS_FUNCTION);

    let gid = load_gid_x(&mut m, &b);
    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, c_total]);
    m.emit_selection_merge(lbl_merge);
    m.emit_branch_conditional(cond, lbl_body, lbl_merge);

    m.emit_label(lbl_body);

    // bh = gid / seq_q, sq_val = gid % seq_q
    let bh = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, bh, gid, c_sq]);
    let sq_val = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, sq_val, gid, c_sq]);
    // q_base = o_base = gid * head_dim
    let q_base = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, q_base, gid, c_hd]);
    // bh_skv = bh * seq_kv  (shared prefix for k/v base)
    let bh_skv = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, bh_skv, bh, c_skv]);

    // ── Pass 1: find max score ──
    m.emit_store(var_max, c_neg_inf);
    m.emit_store(var_sk, b.c_uint_0);
    m.emit_branch(lbl_s1h);

    m.emit_label(lbl_s1h);
    let sk1 = m.alloc_id();
    m.emit_load(b.ty_uint, sk1, var_sk);
    let s1_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, s1_ok, sk1, c_skv]);
    m.emit_loop_merge(lbl_s1m, lbl_s1c);
    m.emit_branch_conditional(s1_ok, lbl_s1b, lbl_s1m);

    m.emit_label(lbl_s1b);
    // Causal check: skip if sq_val < sk1
    if causal {
        let skip = m.alloc_id();
        m.emit(OP_U_LESS_THAN, &[b.ty_bool, skip, sq_val, sk1]);
        m.emit_selection_merge(lbl_s1wm);
        m.emit_branch_conditional(skip, lbl_s1wm, lbl_s1w);
    } else {
        m.emit_branch(lbl_s1w);
    }

    m.emit_label(lbl_s1w);
    // k_base = (bh_skv + sk1) * head_dim
    let ks1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ks1, bh_skv, sk1]);
    let kb1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kb1, ks1, c_hd]);
    // dot product loop
    m.emit_store(var_dot, b.c_float_0);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d1h);

    m.emit_label(lbl_d1h);
    let d1 = m.alloc_id();
    m.emit_load(b.ty_uint, d1, var_d);
    let d1_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d1_ok, d1, c_hd]);
    m.emit_loop_merge(lbl_d1m, lbl_d1c);
    m.emit_branch_conditional(d1_ok, lbl_d1b, lbl_d1m);

    m.emit_label(lbl_d1b);
    let qi1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, qi1, q_base, d1]);
    let qp1 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, qp1, q_var, &[b.c_uint_0, qi1]);
    let qv1 = m.alloc_id();
    m.emit_load(b.ty_float, qv1, qp1);
    let ki1 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ki1, kb1, d1]);
    let kp1 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, kp1, k_var, &[b.c_uint_0, ki1]);
    let kv1 = m.alloc_id();
    m.emit_load(b.ty_float, kv1, kp1);
    let p1 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, p1, qv1, kv1]);
    let od1 = m.alloc_id();
    m.emit_load(b.ty_float, od1, var_dot);
    let nd1 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nd1, od1, p1]);
    m.emit_store(var_dot, nd1);
    m.emit_branch(lbl_d1c);

    m.emit_label(lbl_d1c);
    let d1i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d1i, d1, b.c_uint_1]);
    m.emit_store(var_d, d1i);
    m.emit_branch(lbl_d1h);

    m.emit_label(lbl_d1m);
    // score = dot * scale; max_score = fmax(max_score, score)
    let dot1 = m.alloc_id();
    m.emit_load(b.ty_float, dot1, var_dot);
    let scr1 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, scr1, dot1, c_scale]);
    let om1 = m.alloc_id();
    m.emit_load(b.ty_float, om1, var_max);
    let nm1 = m.alloc_id();
    m.emit_glsl_ext(b.glsl_ext, b.ty_float, nm1, GLSL_F_MAX, &[om1, scr1]);
    m.emit_store(var_max, nm1);

    m.emit_branch(lbl_s1wm);
    m.emit_label(lbl_s1wm);
    m.emit_branch(lbl_s1c);

    m.emit_label(lbl_s1c);
    let sk1i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, sk1i, sk1, b.c_uint_1]);
    m.emit_store(var_sk, sk1i);
    m.emit_branch(lbl_s1h);

    m.emit_label(lbl_s1m);

    // ── Zero output ──
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_zh);

    m.emit_label(lbl_zh);
    let zd = m.alloc_id();
    m.emit_load(b.ty_uint, zd, var_d);
    let zd_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, zd_ok, zd, c_hd]);
    m.emit_loop_merge(lbl_zm, lbl_zc);
    m.emit_branch_conditional(zd_ok, lbl_zb, lbl_zm);

    m.emit_label(lbl_zb);
    let ozi = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ozi, q_base, zd]); // o_base == q_base
    let ozp = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, ozp, o_var, &[b.c_uint_0, ozi]);
    m.emit_store(ozp, b.c_float_0);
    m.emit_branch(lbl_zc);

    m.emit_label(lbl_zc);
    let zdi = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, zdi, zd, b.c_uint_1]);
    m.emit_store(var_d, zdi);
    m.emit_branch(lbl_zh);

    m.emit_label(lbl_zm);

    // ── Pass 2: accumulate weighted V ──
    m.emit_store(var_sum, b.c_float_0);
    m.emit_store(var_sk, b.c_uint_0);
    m.emit_branch(lbl_s2h);

    m.emit_label(lbl_s2h);
    let sk2 = m.alloc_id();
    m.emit_load(b.ty_uint, sk2, var_sk);
    let s2_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, s2_ok, sk2, c_skv]);
    m.emit_loop_merge(lbl_s2m, lbl_s2c);
    m.emit_branch_conditional(s2_ok, lbl_s2b, lbl_s2m);

    m.emit_label(lbl_s2b);
    if causal {
        let skip2 = m.alloc_id();
        m.emit(OP_U_LESS_THAN, &[b.ty_bool, skip2, sq_val, sk2]);
        m.emit_selection_merge(lbl_s2wm);
        m.emit_branch_conditional(skip2, lbl_s2wm, lbl_s2w);
    } else {
        m.emit_branch(lbl_s2w);
    }

    m.emit_label(lbl_s2w);
    // k/v base
    let ks2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ks2, bh_skv, sk2]);
    let kb2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, kb2, ks2, c_hd]);
    // dot product d2 loop
    m.emit_store(var_dot, b.c_float_0);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d2h);

    m.emit_label(lbl_d2h);
    let d2 = m.alloc_id();
    m.emit_load(b.ty_uint, d2, var_d);
    let d2_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d2_ok, d2, c_hd]);
    m.emit_loop_merge(lbl_d2m, lbl_d2c);
    m.emit_branch_conditional(d2_ok, lbl_d2b, lbl_d2m);

    m.emit_label(lbl_d2b);
    let qi2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, qi2, q_base, d2]);
    let qp2 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, qp2, q_var, &[b.c_uint_0, qi2]);
    let qv2 = m.alloc_id();
    m.emit_load(b.ty_float, qv2, qp2);
    let ki2 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, ki2, kb2, d2]);
    let kp2 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, kp2, k_var, &[b.c_uint_0, ki2]);
    let kv2 = m.alloc_id();
    m.emit_load(b.ty_float, kv2, kp2);
    let p2 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, p2, qv2, kv2]);
    let od2 = m.alloc_id();
    m.emit_load(b.ty_float, od2, var_dot);
    let nd2 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nd2, od2, p2]);
    m.emit_store(var_dot, nd2);
    m.emit_branch(lbl_d2c);

    m.emit_label(lbl_d2c);
    let d2i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d2i, d2, b.c_uint_1]);
    m.emit_store(var_d, d2i);
    m.emit_branch(lbl_d2h);

    m.emit_label(lbl_d2m);
    // w = exp(dot*scale − max_score); sum_exp += w
    let dot2 = m.alloc_id();
    m.emit_load(b.ty_float, dot2, var_dot);
    let scr2 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, scr2, dot2, c_scale]);
    let mx2 = m.alloc_id();
    m.emit_load(b.ty_float, mx2, var_max);
    let diff = m.alloc_id();
    m.emit(OP_F_SUB, &[b.ty_float, diff, scr2, mx2]);
    let w = m.alloc_id();
    m.emit_glsl_ext(b.glsl_ext, b.ty_float, w, GLSL_EXP, &[diff]);
    let os2 = m.alloc_id();
    m.emit_load(b.ty_float, os2, var_sum);
    let ns2 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, ns2, os2, w]);
    m.emit_store(var_sum, ns2);

    // V accumulation d3 loop: o[o_base+d] += w * v[kb2+d]
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d3h);

    m.emit_label(lbl_d3h);
    let d3 = m.alloc_id();
    m.emit_load(b.ty_uint, d3, var_d);
    let d3_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d3_ok, d3, c_hd]);
    m.emit_loop_merge(lbl_d3m, lbl_d3c);
    m.emit_branch_conditional(d3_ok, lbl_d3b, lbl_d3m);

    m.emit_label(lbl_d3b);
    let vi3 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, vi3, kb2, d3]);
    let vp3 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, vp3, v_var, &[b.c_uint_0, vi3]);
    let vv3 = m.alloc_id();
    m.emit_load(b.ty_float, vv3, vp3);
    let wv3 = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, wv3, w, vv3]);
    let oi3 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oi3, q_base, d3]);
    let op3 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, op3, o_var, &[b.c_uint_0, oi3]);
    let ov3 = m.alloc_id();
    m.emit_load(b.ty_float, ov3, op3);
    let nv3 = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, nv3, ov3, wv3]);
    m.emit_store(op3, nv3);
    m.emit_branch(lbl_d3c);

    m.emit_label(lbl_d3c);
    let d3i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d3i, d3, b.c_uint_1]);
    m.emit_store(var_d, d3i);
    m.emit_branch(lbl_d3h);

    m.emit_label(lbl_d3m);
    m.emit_branch(lbl_s2wm);

    m.emit_label(lbl_s2wm);
    m.emit_branch(lbl_s2c);

    m.emit_label(lbl_s2c);
    let sk2i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, sk2i, sk2, b.c_uint_1]);
    m.emit_store(var_sk, sk2i);
    m.emit_branch(lbl_s2h);

    m.emit_label(lbl_s2m);

    // ── Normalize: o[o_base+d] /= sum_exp ──
    let final_sum = m.alloc_id();
    m.emit_load(b.ty_float, final_sum, var_sum);
    m.emit_store(var_d, b.c_uint_0);
    m.emit_branch(lbl_d4h);

    m.emit_label(lbl_d4h);
    let d4 = m.alloc_id();
    m.emit_load(b.ty_uint, d4, var_d);
    let d4_ok = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, d4_ok, d4, c_hd]);
    m.emit_loop_merge(lbl_d4m, lbl_d4c);
    m.emit_branch_conditional(d4_ok, lbl_d4b, lbl_d4m);

    m.emit_label(lbl_d4b);
    let oi4 = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, oi4, q_base, d4]);
    let op4 = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, op4, o_var, &[b.c_uint_0, oi4]);
    let ov4 = m.alloc_id();
    m.emit_load(b.ty_float, ov4, op4);
    let nv4 = m.alloc_id();
    m.emit(OP_F_DIV, &[b.ty_float, nv4, ov4, final_sum]);
    m.emit_store(op4, nv4);
    m.emit_branch(lbl_d4c);

    m.emit_label(lbl_d4c);
    let d4i = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, d4i, d4, b.c_uint_1]);
    m.emit_store(var_d, d4i);
    m.emit_branch(lbl_d4h);

    m.emit_label(lbl_d4m);
    m.emit_branch(lbl_merge);

    m.emit_label(lbl_merge);
    m.emit_return();
    m.emit_function_end();
    m.finalize()
}
