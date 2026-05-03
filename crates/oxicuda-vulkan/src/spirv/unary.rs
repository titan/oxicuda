//! SPIR-V compute-shader generator for element-wise unary operations.

use oxicuda_backend::UnaryOp;

use super::builder::SpvModule;
use super::consts::{
    FUNCTION_CONTROL_NONE, GLSL_EXP, GLSL_F_ABS, GLSL_F_MAX, GLSL_LOG, GLSL_SQRT, GLSL_TANH,
    OP_F_ADD, OP_F_DIV, OP_F_NEGATE, OP_U_LESS_THAN, SPIRV_VERSION_1_3,
};
use super::preamble::{
    BaseIds, emit_float_ssbo, emit_preamble, emit_uint_ssbo, load_gid_x, load_param_uint,
};

/// Generate a SPIR-V compute shader for an element-wise unary operation.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`, 2 = params `uint[]`
/// where `params[0] = count`.
pub fn unary_compute_shader(op: UnaryOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);
    let b = emit_preamble(&mut m);

    let (_, _, input_var) = emit_float_ssbo(&mut m, &b, 0);
    let (_, _, output_var) = emit_float_ssbo(&mut m, &b, 1);
    let (_, _, params_var) = emit_uint_ssbo(&mut m, &b, 2);

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

    let inp_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, inp_ptr, input_var, &[b.c_uint_0, gid]);
    let inp_val = m.alloc_id();
    m.emit_load(b.ty_float, inp_val, inp_ptr);

    let result = emit_unary_op(&mut m, &b, op, inp_val);

    let out_ptr = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_float, out_ptr, output_var, &[b.c_uint_0, gid]);
    m.emit_store(out_ptr, result);

    m.emit_branch(label_merge);

    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

/// Emit the SPIR-V instructions for a unary operation, returning the result ID.
fn emit_unary_op(m: &mut SpvModule, b: &BaseIds, op: UnaryOp, x: u32) -> u32 {
    let result = m.alloc_id();
    match op {
        UnaryOp::Relu => {
            m.emit_glsl_ext(
                b.glsl_ext,
                b.ty_float,
                result,
                GLSL_F_MAX,
                &[b.c_float_0, x],
            );
        }
        UnaryOp::Sigmoid => {
            let neg_x = m.alloc_id();
            m.emit(OP_F_NEGATE, &[b.ty_float, neg_x, x]);
            let exp_neg_x = m.alloc_id();
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, exp_neg_x, GLSL_EXP, &[neg_x]);
            let one_plus = m.alloc_id();
            m.emit(OP_F_ADD, &[b.ty_float, one_plus, b.c_float_1, exp_neg_x]);
            m.emit(OP_F_DIV, &[b.ty_float, result, b.c_float_1, one_plus]);
        }
        UnaryOp::Tanh => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_TANH, &[x]);
        }
        UnaryOp::Exp => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_EXP, &[x]);
        }
        UnaryOp::Log => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_LOG, &[x]);
        }
        UnaryOp::Sqrt => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_SQRT, &[x]);
        }
        UnaryOp::Abs => {
            m.emit_glsl_ext(b.glsl_ext, b.ty_float, result, GLSL_F_ABS, &[x]);
        }
        UnaryOp::Neg => {
            m.emit(OP_F_NEGATE, &[b.ty_float, result, x]);
        }
    }
    result
}
