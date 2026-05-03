//! Common SPIR-V preamble helpers shared by every compute-shader generator.
//!
//! Provides the [`BaseIds`] bundle and the convenience helpers that emit
//! the boiler-plate prologue (capabilities, extension imports, base types,
//! constants, GlobalInvocationId variable) consumed by every shader in
//! this module tree.

use super::builder::SpvModule;
use super::consts::{
    BUILTIN_GLOBAL_INVOCATION_ID, CAPABILITY_SHADER, DECORATION_ARRAY_STRIDE, DECORATION_BINDING,
    DECORATION_BLOCK, DECORATION_BUILTIN, DECORATION_DESCRIPTOR_SET, DECORATION_OFFSET,
    STORAGE_CLASS_FUNCTION, STORAGE_CLASS_INPUT, STORAGE_CLASS_STORAGE_BUFFER, WORKGROUP_SIZE,
};

/// IDs shared by all compute shaders.
pub(super) struct BaseIds {
    pub(super) ty_void: u32,
    pub(super) ty_bool: u32,
    pub(super) ty_uint: u32,
    pub(super) ty_float: u32,
    #[allow(dead_code)]
    pub(super) ty_v3uint: u32,
    pub(super) ty_fn_void: u32,
    #[allow(dead_code)]
    pub(super) ty_ptr_input_v3uint: u32,
    pub(super) ty_ptr_input_uint: u32,
    pub(super) ty_rt_array_float: u32,
    #[allow(dead_code)]
    pub(super) ty_rt_array_uint: u32,
    pub(super) ty_ptr_sb_float: u32,
    pub(super) ty_ptr_sb_uint: u32,
    pub(super) ty_ptr_func_float: u32,
    pub(super) ty_ptr_func_uint: u32,
    pub(super) c_uint_0: u32,
    pub(super) c_uint_1: u32,
    pub(super) c_float_0: u32,
    pub(super) c_float_1: u32,
    pub(super) var_gid: u32,
    pub(super) glsl_ext: u32,
    pub(super) main_fn: u32,
}

/// Emit the preamble shared by all compute shaders and return [BaseIds].
pub(super) fn emit_preamble(m: &mut SpvModule) -> BaseIds {
    let main_fn = m.alloc_id();
    let ty_void = m.alloc_id();
    let ty_bool = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_fn_void = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_input_uint = m.alloc_id();
    let ty_rt_array_float = m.alloc_id();
    let ty_rt_array_uint = m.alloc_id();
    let ty_ptr_sb_float = m.alloc_id();
    let ty_ptr_sb_uint = m.alloc_id();
    let ty_ptr_func_float = m.alloc_id();
    let ty_ptr_func_uint = m.alloc_id();
    let c_uint_0 = m.alloc_id();
    let c_uint_1 = m.alloc_id();
    let c_float_0 = m.alloc_id();
    let c_float_1 = m.alloc_id();
    let var_gid = m.alloc_id();
    let glsl_ext = m.alloc_id();

    m.emit_capability(CAPABILITY_SHADER);
    m.emit_ext_inst_import(glsl_ext, "GLSL.std.450");
    m.emit_memory_model();
    m.emit_entry_point(main_fn, "main", &[var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(ty_rt_array_float, DECORATION_ARRAY_STRIDE, &[4]);
    m.emit_decorate(ty_rt_array_uint, DECORATION_ARRAY_STRIDE, &[4]);

    m.emit_type_void(ty_void);
    m.emit_type_bool(ty_bool);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_function(ty_fn_void, ty_void, &[]);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_input_uint, STORAGE_CLASS_INPUT, ty_uint);
    m.emit_type_runtime_array(ty_rt_array_float, ty_float);
    m.emit_type_runtime_array(ty_rt_array_uint, ty_uint);
    m.emit_type_pointer(ty_ptr_sb_float, STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    m.emit_type_pointer(ty_ptr_sb_uint, STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    m.emit_type_pointer(ty_ptr_func_float, STORAGE_CLASS_FUNCTION, ty_float);
    m.emit_type_pointer(ty_ptr_func_uint, STORAGE_CLASS_FUNCTION, ty_uint);

    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_uint_1, 1);
    m.emit_constant_f32(ty_float, c_float_0, 0.0);
    m.emit_constant_f32(ty_float, c_float_1, 1.0);

    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);

    BaseIds {
        ty_void,
        ty_bool,
        ty_uint,
        ty_float,
        ty_v3uint,
        ty_fn_void,
        ty_ptr_input_v3uint,
        ty_ptr_input_uint,
        ty_rt_array_float,
        ty_rt_array_uint,
        ty_ptr_sb_float,
        ty_ptr_sb_uint,
        ty_ptr_func_float,
        ty_ptr_func_uint,
        c_uint_0,
        c_uint_1,
        c_float_0,
        c_float_1,
        var_gid,
        glsl_ext,
        main_fn,
    }
}

/// Emit a float SSBO and return `(struct_type, ptr_type, variable)`.
pub(super) fn emit_float_ssbo(m: &mut SpvModule, b: &BaseIds, binding: u32) -> (u32, u32, u32) {
    let struct_ty = m.alloc_id();
    let ptr_ty = m.alloc_id();
    let var = m.alloc_id();

    m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(var, DECORATION_BINDING, &[binding]);

    m.emit_type_struct(struct_ty, &[b.ty_rt_array_float]);
    m.emit_type_pointer(ptr_ty, STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
    m.emit_variable(ptr_ty, var, STORAGE_CLASS_STORAGE_BUFFER);

    (struct_ty, ptr_ty, var)
}

/// Emit a uint SSBO (for params) and return `(struct_type, ptr_type, variable)`.
pub(super) fn emit_uint_ssbo(m: &mut SpvModule, b: &BaseIds, binding: u32) -> (u32, u32, u32) {
    let struct_ty = m.alloc_id();
    let ptr_ty = m.alloc_id();
    let var = m.alloc_id();

    m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
    m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
    m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
    m.emit_decorate(var, DECORATION_BINDING, &[binding]);

    m.emit_type_struct(struct_ty, &[b.ty_rt_array_uint]);
    m.emit_type_pointer(ptr_ty, STORAGE_CLASS_STORAGE_BUFFER, struct_ty);
    m.emit_variable(ptr_ty, var, STORAGE_CLASS_STORAGE_BUFFER);

    (struct_ty, ptr_ty, var)
}

/// Load `GlobalInvocationId.x` into a uint result.
pub(super) fn load_gid_x(m: &mut SpvModule, b: &BaseIds) -> u32 {
    let ptr = m.alloc_id();
    let gid = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_input_uint, ptr, b.var_gid, &[b.c_uint_0]);
    m.emit_load(b.ty_uint, gid, ptr);
    gid
}

/// Load a uint from params SSBO at constant `idx_const`.
pub(super) fn load_param_uint(
    m: &mut SpvModule,
    b: &BaseIds,
    params_var: u32,
    idx_const: u32,
) -> u32 {
    let ptr = m.alloc_id();
    let val = m.alloc_id();
    m.emit_access_chain(b.ty_ptr_sb_uint, ptr, params_var, &[b.c_uint_0, idx_const]);
    m.emit_load(b.ty_uint, val, ptr);
    val
}
