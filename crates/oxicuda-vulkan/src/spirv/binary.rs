//! SPIR-V compute-shader generator for element-wise binary operations.

use oxicuda_backend::BinaryOp;

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, GLSL_F_MAX, GLSL_F_MIN, OP_F_ADD, OP_F_DIV, OP_F_MUL, OP_F_SUB,
    OP_U_LESS_THAN, SPIRV_VERSION_1_3,
};
use super::preamble::{
    BaseIds, emit_float_ssbo, emit_preamble, emit_uint_ssbo, load_gid_x, load_param_uint,
};

/// Generate a SPIR-V compute shader for an element-wise binary operation.
///
/// Bindings: 0 = a `float[]`, 1 = b `float[]`, 2 = output `float[]`,
/// 3 = params `uint[]` where `params[0] = count`.
pub fn binary_compute_shader(op: BinaryOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, a_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, b_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, out_var) = emit_float_ssbo(&mut m, &b, 2);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 3);

    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(b.ty_void, b.main_fn, FUNCTION_CONTROL_NONE, b.ty_fn_void);
    m.emit_label(label_entry);

    let gid = load_gid_x(&mut m, &b);
    let count = load_param_uint(&mut m, &b, params_var, b.c_uint_0);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[b.ty_bool, cond, gid, count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    let a_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, a_ptr, a_var, &[b.c_uint_0, gid]);
    let a_val = m.alloc_id();
    m.emit_load(b.ty_float, a_val, a_ptr);

    let b_ptr_id = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, b_ptr_id, b_var, &[b.c_uint_0, gid]);
    let b_val = m.alloc_id();
    m.emit_load(b.ty_float, b_val, b_ptr_id);

    let result = emit_binary_op(&mut m, &b, op, a_val, b_val);

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, out_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, result);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

fn emit_binary_op(m: &mut SpvModule, b: &BaseIds, op: BinaryOp, lhs: u32, rhs: u32) -> u32 {
    let result = m.alloc_id();
    match op {
        BinaryOp::Add => m.emit(OP_F_ADD, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Sub => m.emit(OP_F_SUB, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Mul => m.emit(OP_F_MUL, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Div => m.emit(OP_F_DIV, &[b.ty_float, result, lhs, rhs]),
        BinaryOp::Max => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_MAX, &[lhs, rhs]);
        }
        BinaryOp::Min => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_MIN, &[lhs, rhs]);
        }
    }
    result
}
