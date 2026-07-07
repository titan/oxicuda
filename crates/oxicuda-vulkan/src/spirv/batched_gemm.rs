//! SPIR-V compute-shader generator for strided batched GEMM.

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, OP_BITCAST, OP_F_ADD, OP_F_MUL, OP_I_ADD, OP_I_MUL, OP_U_DIV,
    OP_U_LESS_THAN, OP_U_MOD, SPIRV_VERSION_1_3, STORAGE_CLASS_FUNCTION,
};
use super::preamble::{
    emit_float_ssbo, emit_preamble, emit_uint_ssbo, load_gid_x, load_param_uint,
};

/// Generate a SPIR-V compute shader for strided batched GEMM.
///
/// For each batch index `b` (from `GlobalInvocationId.z`), computes
/// `C_b = alpha * A_b * B_b + beta * C_b` where the batch matrices are
/// offset by `stride_a`, `stride_b`, `stride_c` elements respectively.
///
/// Bindings: 0 = A `float[]`, 1 = B `float[]`, 2 = C `float[]`,
/// 3 = params `uint[]` where:
///   `params[0..5]` = `[m, n, k, alpha(bitcast), beta(bitcast)]`
///   `params[5..8]` = `[stride_a, stride_b, stride_c]`
///
/// Dispatch: `(ceil(m*n / 256), 1, batch_count)`.
pub fn batched_gemm_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, a_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, b_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, c_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 3);

    // Additional uint constants for param indices 2..7
    let c_uint_2 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_2, 2);
    let c_uint_3 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_3, 3);
    let c_uint_4 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_4, 4);
    let c_uint_5 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_5, 5);
    let c_uint_6 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_6, 6);
    let c_uint_7 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_7, 7);

    // Labels
    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    // SPIR-V requires all Function-storage OpVariables to be declared in the
    // first (entry) basic block, so allocate them here before any branch.
    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);

    // Load GlobalInvocationId.x and .z
    let gid_x = load_gid_x(&mut m, &b);
    // Load GlobalInvocationId.z (batch index)
    let gid_z_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_input_uint, gid_z_ptr, b.var_gid, &[c_uint_2]);
    let gid_z = m.alloc_id();
    m.emit_load(b.ty_uint, gid_z, gid_z_ptr);

    // Load params
    let param_m = load_param_uint(&mut m, &b, params_var, b.c_uint_0);
    let param_n = load_param_uint(&mut m, &b, params_var, b.c_uint_1);
    let param_k = load_param_uint(&mut m, &b, params_var, c_uint_2);

    let alpha_u = load_param_uint(&mut m, &b, params_var, c_uint_3);
    let alpha = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, alpha, alpha_u]);
    let beta_u = load_param_uint(&mut m, &b, params_var, c_uint_4);
    let beta = m.alloc_id();
    m.emit(OP_BITCAST, &[b.ty_float, beta, beta_u]);

    let p_stride_a = load_param_uint(&mut m, &b, params_var, c_uint_5);
    let p_stride_b = load_param_uint(&mut m, &b, params_var, c_uint_6);
    let p_stride_c = load_param_uint(&mut m, &b, params_var, c_uint_7);

    // Bounds check: gid_x < m * n
    let total = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total, param_m, param_n]);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid_x, total]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    // Compute batch offsets: base_a = gid_z * stride_a, etc.
    let base_a = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_a, gid_z, p_stride_a]);
    let base_b = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_b, gid_z, p_stride_b]);
    let base_c = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, base_c, gid_z, p_stride_c]);

    // row = gid_x / n, col = gid_x % n
    let row = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, row, gid_x, param_n]);
    let col = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, col, gid_x, param_n]);

    // Initialise the loop counter and accumulator (declared in the entry block).
    m.emit_store(var_i, b.c_uint_0);
    m.emit_store(var_acc, b.c_float_0);

    m.emit_branch(label_loop_header);

    // Loop header
    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, param_k]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    // Loop body
    m.emit_label(label_loop_body);

    // a_idx = base_a + row * k + i
    let row_k = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, row_k, row, param_k]);
    let a_local = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_local, row_k, i_val]);
    let a_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, a_idx, base_a, a_local]);

    // b_idx = base_b + i * n + col
    let i_n = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_n, i_val, param_n]);
    let b_local = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_local, i_n, col]);
    let b_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, b_idx, base_b, b_local]);

    let a_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, a_ptr, a_var, &[b.c_uint_0, a_idx]);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr_id = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, b_ptr_id, b_var, &[b.c_uint_0, b_idx]);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr_id);

    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, prod, a_val, b_val]);
    let old_acc = m.alloc_id();
    m.emit_load(b.ty_float, old_acc, var_acc);
    let new_acc = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, new_acc, old_acc, prod]);
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    // Loop continue
    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    // Loop merge — compute result = alpha * acc + beta * C[c_idx]
    m.emit_label(label_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);
    let alpha_acc = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, alpha_acc, alpha, final_acc]);

    // c_idx = base_c + gid_x
    let c_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, c_idx, base_c, gid_x]);

    let c_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, c_ptr, c_var, &[b.c_uint_0, c_idx]);
    let c_old = m.alloc_id();
    m.emit_load(b.ty_float, c_old, c_ptr);
    let beta_c = m.alloc_id();
    m.emit(OP_F_MUL, &[b.ty_float, beta_c, beta, c_old]);
    let c_new = m.alloc_id();
    m.emit(OP_F_ADD, &[b.ty_float, c_new, alpha_acc, beta_c]);
    m.emit_store(c_ptr, c_new);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}
