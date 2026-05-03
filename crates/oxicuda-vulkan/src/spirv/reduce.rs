//! SPIR-V compute-shader generator for axis-aligned reductions.

use oxicuda_backend::ReduceOp;

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, GLSL_F_MAX, GLSL_F_MIN, OP_CONVERT_U_TO_F, OP_F_ADD, OP_F_DIV, OP_I_ADD,
    OP_I_MUL, OP_U_DIV, OP_U_LESS_THAN, OP_U_MOD, SPIRV_VERSION_1_3, STORAGE_CLASS_FUNCTION,
};
use super::preamble::{
    emit_float_ssbo, emit_preamble, emit_uint_ssbo, load_gid_x, load_param_uint,
};

/// Generate a SPIR-V compute shader for reduction along an axis.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`,
/// 2 = params `uint[]` where params = [outer_size, reduce_size, inner_size].
pub fn reduce_compute_shader(op: ReduceOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 2);

    let c_uint_2 = m.alloc_id();
    m.emit_constant_u32(b.ty_uint, c_uint_2, 2);

    let init_val = match op {
        ReduceOp::Sum | ReduceOp::Mean => b.c_float_0,
        ReduceOp::Max => {
            let neg_inf = m.alloc_id();
            m.emit_constant_f32(b.ty_float, neg_inf, f32::NEG_INFINITY);
            neg_inf
        }
        ReduceOp::Min => {
            let pos_inf = m.alloc_id();
            m.emit_constant_f32(b.ty_float, pos_inf, f32::INFINITY);
            pos_inf
        }
    };

    let label_entry = m.alloc_id();
    let label_bounds_body = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_loop_header = m.alloc_id();
    let label_loop_body = m.alloc_id();
    let label_loop_continue = m.alloc_id();
    let label_loop_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);

    let outer_size = load_param_uint(&mut m, &b, params_var, b.c_uint_0);
    let reduce_size = load_param_uint(&mut m, &b, params_var, b.c_uint_1);
    let inner_size = load_param_uint(&mut m, &b, params_var, c_uint_2);

    let total_output = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, total_output, outer_size, inner_size]);

    let cond_bounds = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond_bounds, gid, total_output]);
    m.emit_selection_merge(label_bounds_merge);
    m.emit_branch_conditional(cond_bounds, label_bounds_body, label_bounds_merge);

    m.emit_label(label_bounds_body);

    let outer_idx = m.alloc_id();
    m.emit(OP_U_DIV, &[b.ty_uint, outer_idx, gid, inner_size]);
    let inner_idx = m.alloc_id();
    m.emit(OP_U_MOD, &[b.ty_uint, inner_idx, gid, inner_size]);

    // base = outer_idx * reduce_size * inner_size + inner_idx
    let t1 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t1, outer_idx, reduce_size]);
    let t2 = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, t2, t1, inner_size]);
    let base_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, base_idx, t2, inner_idx]);

    let var_i = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_uint, var_i, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_i, b.c_uint_0);

    let var_acc = m.alloc_id();
    m.emit_variable(b.ty_ptr_func_float, var_acc, STORAGE_CLASS_FUNCTION);
    m.emit_store(var_acc, init_val);

    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_header);
    let i_val = m.alloc_id();
    m.emit_load(b.ty_uint, i_val, var_i);
    let loop_cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, loop_cond, i_val, reduce_size]);
    m.emit_loop_merge(label_loop_merge, label_loop_continue);
    m.emit_branch_conditional(loop_cond, label_loop_body, label_loop_merge);

    m.emit_label(label_loop_body);

    // input_idx = base_idx + i * inner_size
    let i_times_inner = m.alloc_id();
    m.emit(OP_I_MUL, &[b.ty_uint, i_times_inner, i_val, inner_size]);
    let input_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, input_idx, base_idx, i_times_inner]);

    let inp_ptr = m.alloc_id();
    m.emit_access_chain(
        b.ty_ptr_sb_float,
        inp_ptr,
        input_var,
        &[b.c_uint_0, input_idx],
    );
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    let acc_val = m.alloc_id();
    m.emit_load(b.ty_float, acc_val, var_acc);

    let new_acc = m.alloc_id();
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            m.emit(OP_F_ADD, &[b.ty_float, new_acc, acc_val, inp_val]);
        }
        ReduceOp::Max => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                new_acc,
                GLSL_F_MAX,
                &[acc_val, inp_val],
            );
        }
        ReduceOp::Min => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                new_acc,
                GLSL_F_MIN,
                &[acc_val, inp_val],
            );
        }
    }
    m.emit_store(var_acc, new_acc);

    m.emit_branch(label_loop_continue);

    m.emit_label(label_loop_continue);
    let i_inc = m.alloc_id();
    m.emit(OP_I_ADD, &[b.ty_uint, i_inc, i_val, b.c_uint_1]);
    m.emit_store(var_i, i_inc);
    m.emit_branch(label_loop_header);

    m.emit_label(label_loop_merge);

    let final_acc = m.alloc_id();
    m.emit_load(b.ty_float, final_acc, var_acc);

    let store_val = if op == ReduceOp::Mean {
        let reduce_f = m.alloc_id();
        m.emit(OP_CONVERT_U_TO_F, &[b.ty_float, reduce_f, reduce_size]);
        let mean_val = m.alloc_id();
        m.emit(OP_F_DIV, &[b.ty_float, mean_val, final_acc, reduce_f]);
        mean_val
    } else {
        final_acc
    };

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, store_val);

    m.emit_branch(label_bounds_merge);

    m.emit_label(label_bounds_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}
