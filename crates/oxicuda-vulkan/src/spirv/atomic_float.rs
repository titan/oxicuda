//! Single-pass FP32 atomic-reduction SPIR-V generator (`VK_EXT_shader_atomic_float`).
//!
//! On hardware that advertises `shaderBufferFloat32AtomicAdd` (and the
//! `AtomicFloat32AddEXT` SPIR-V capability) a reduction can be performed in a
//! single dispatch: every invocation atomically accumulates its element into a
//! single output slot, removing the second pass required by the shared-memory
//! tree reduction.
//!
//! Bindings: 0 = input `float[]`, 1 = output `float[]` (slot 0 is the
//! accumulator), 2 = params `uint[]` with `params[0] = count`.
//!
//! The host must pre-initialise `output[0]` to the reduction identity
//! (`0` for add, `+inf` for min, `-inf` for max) before dispatch.
//!
//! Capabilities are emitted *before* the memory model so the module respects
//! the SPIR-V logical layout; the shader is therefore built directly rather
//! than via the shared preamble (which already commits the memory model).

use super::builder::SpvModule;
use super::consts::{
    BUILTIN_GLOBAL_INVOCATION_ID, CAPABILITY_ATOMIC_FLOAT32_ADD_EXT,
    CAPABILITY_ATOMIC_FLOAT32_MIN_MAX_EXT, CAPABILITY_SHADER, DECORATION_ARRAY_STRIDE,
    DECORATION_BINDING, DECORATION_BLOCK, DECORATION_BUILTIN, DECORATION_DESCRIPTOR_SET,
    DECORATION_OFFSET, FUNCTION_CONTROL_NONE, MEMORY_SEMANTICS_ACQUIRE_RELEASE,
    OP_ATOMIC_F_ADD_EXT, OP_ATOMIC_F_MAX_EXT, OP_ATOMIC_F_MIN_EXT, OP_U_LESS_THAN, SCOPE_DEVICE,
    SPIRV_VERSION_1_3, STORAGE_CLASS_INPUT, STORAGE_CLASS_STORAGE_BUFFER, WORKGROUP_SIZE,
};

/// The atomic-float reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicFloatOp {
    /// Atomic floating-point sum.
    Add,
    /// Atomic floating-point minimum.
    Min,
    /// Atomic floating-point maximum.
    Max,
}

impl AtomicFloatOp {
    fn opcode(self) -> u32 {
        match self {
            AtomicFloatOp::Add => OP_ATOMIC_F_ADD_EXT,
            AtomicFloatOp::Min => OP_ATOMIC_F_MIN_EXT,
            AtomicFloatOp::Max => OP_ATOMIC_F_MAX_EXT,
        }
    }

    fn capability(self) -> u32 {
        match self {
            AtomicFloatOp::Add => CAPABILITY_ATOMIC_FLOAT32_ADD_EXT,
            AtomicFloatOp::Min | AtomicFloatOp::Max => CAPABILITY_ATOMIC_FLOAT32_MIN_MAX_EXT,
        }
    }
}

/// Generate a single-pass FP32 atomic-reduction compute shader.
///
/// Every in-range invocation loads `input[gid]` and folds it into `output[0]`
/// with one atomic float instruction.
#[must_use]
pub fn atomic_float_reduce_spirv(op: AtomicFloatOp) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_3);

    // ── IDs ──
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
    let ty_struct_float = m.alloc_id();
    let ty_struct_uint = m.alloc_id();
    let ty_ptr_sb_struct_float = m.alloc_id();
    let ty_ptr_sb_struct_uint = m.alloc_id();

    let c_uint_0 = m.alloc_id();
    let c_scope_device = m.alloc_id();
    let c_semantics = m.alloc_id();

    let var_gid = m.alloc_id();
    let var_input = m.alloc_id();
    let var_output = m.alloc_id();
    let var_params = m.alloc_id();

    // ── Capabilities (must precede the memory model) ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(op.capability());

    m.emit_memory_model();
    m.emit_entry_point(main_fn, "main", &[var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    m.emit_decorate(ty_rt_array_float, DECORATION_ARRAY_STRIDE, &[4]);
    m.emit_decorate(ty_rt_array_uint, DECORATION_ARRAY_STRIDE, &[4]);
    for (struct_ty, var, binding) in [
        (ty_struct_float, var_input, 0u32),
        (ty_struct_float, var_output, 1u32),
        (ty_struct_uint, var_params, 2u32),
    ] {
        m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
        m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
        m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
        m.emit_decorate(var, DECORATION_BINDING, &[binding]);
    }

    // ── Types ──
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
    m.emit_type_struct(ty_struct_float, &[ty_rt_array_float]);
    m.emit_type_struct(ty_struct_uint, &[ty_rt_array_uint]);
    m.emit_type_pointer(ty_ptr_sb_float, STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    m.emit_type_pointer(ty_ptr_sb_uint, STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    m.emit_type_pointer(
        ty_ptr_sb_struct_float,
        STORAGE_CLASS_STORAGE_BUFFER,
        ty_struct_float,
    );
    m.emit_type_pointer(
        ty_ptr_sb_struct_uint,
        STORAGE_CLASS_STORAGE_BUFFER,
        ty_struct_uint,
    );

    // ── Constants ──
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_scope_device, SCOPE_DEVICE);
    m.emit_constant_u32(ty_uint, c_semantics, MEMORY_SEMANTICS_ACQUIRE_RELEASE);

    // ── Variables ──
    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);
    m.emit_variable(
        ty_ptr_sb_struct_float,
        var_input,
        STORAGE_CLASS_STORAGE_BUFFER,
    );
    m.emit_variable(
        ty_ptr_sb_struct_float,
        var_output,
        STORAGE_CLASS_STORAGE_BUFFER,
    );
    m.emit_variable(
        ty_ptr_sb_struct_uint,
        var_params,
        STORAGE_CLASS_STORAGE_BUFFER,
    );

    // ── Function body ──
    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn_void);
    m.emit_label(label_entry);

    // gid = GlobalInvocationId.x
    let gid_ptr = m.alloc_id();
    let gid = m.alloc_id();
    m.emit_access_chain(ty_ptr_input_uint, gid_ptr, var_gid, &[c_uint_0]);
    m.emit_load(ty_uint, gid, gid_ptr);

    // count = params[0]
    let cnt_ptr = m.alloc_id();
    let count = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_uint, cnt_ptr, var_params, &[c_uint_0, c_uint_0]);
    m.emit_load(ty_uint, count, cnt_ptr);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond, gid, count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    // value = input[gid]
    let inp_ptr = m.alloc_id();
    let value = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, inp_ptr, var_input, &[c_uint_0, gid]);
    m.emit_load(ty_float, value, inp_ptr);

    // out_ptr = &output[0]
    let out_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, out_ptr, var_output, &[c_uint_0, c_uint_0]);

    // OpAtomicF{Add,Min,Max}EXT <ty> <id> <Pointer> <Scope> <Semantics> <Value>
    let old = m.alloc_id();
    m.emit(
        op.opcode(),
        &[ty_float, old, out_ptr, c_scope_device, c_semantics, value],
    );

    m.emit_branch(label_merge);
    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::consts::{
        OP_ATOMIC_F_ADD_EXT, OP_ATOMIC_F_MAX_EXT, OP_ATOMIC_F_MIN_EXT, SPIRV_MAGIC,
        SPIRV_VERSION_1_3,
    };

    fn opcodes(words: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 5usize;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            out.push(words[i] & 0xFFFF);
            i += count;
        }
        out
    }

    fn capabilities(words: &[u32]) -> Vec<u32> {
        let mut caps = Vec::new();
        let mut i = 5usize;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            if (words[i] & 0xFFFF) == 17 {
                caps.push(words[i + 1]);
            }
            i += count;
        }
        caps
    }

    #[test]
    fn header_valid_for_all_ops() {
        for op in [AtomicFloatOp::Add, AtomicFloatOp::Min, AtomicFloatOp::Max] {
            let w = atomic_float_reduce_spirv(op);
            assert_eq!(w[0], SPIRV_MAGIC, "op {op:?}");
            assert_eq!(w[1], SPIRV_VERSION_1_3);
            assert!(w[3] > 0);
            assert_eq!(w[4], 0);
        }
    }

    #[test]
    fn add_emits_atomic_fadd() {
        let w = atomic_float_reduce_spirv(AtomicFloatOp::Add);
        assert!(opcodes(&w).contains(&OP_ATOMIC_F_ADD_EXT));
    }

    #[test]
    fn min_max_emit_correct_atomic_opcodes() {
        assert!(
            opcodes(&atomic_float_reduce_spirv(AtomicFloatOp::Min)).contains(&OP_ATOMIC_F_MIN_EXT)
        );
        assert!(
            opcodes(&atomic_float_reduce_spirv(AtomicFloatOp::Max)).contains(&OP_ATOMIC_F_MAX_EXT)
        );
    }

    #[test]
    fn capabilities_precede_memory_model() {
        // OpMemoryModel == 14. All capabilities (17) must appear before it.
        let w = atomic_float_reduce_spirv(AtomicFloatOp::Add);
        let mut i = 5usize;
        let mut seen_mem_model = false;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            let opcode = w[i] & 0xFFFF;
            if opcode == 14 {
                seen_mem_model = true;
            }
            if opcode == 17 {
                assert!(
                    !seen_mem_model,
                    "capability after memory model violates layout"
                );
            }
            i += count;
        }
        assert!(seen_mem_model, "module must declare a memory model");
    }

    #[test]
    fn add_and_minmax_use_different_capabilities() {
        let add_caps = capabilities(&atomic_float_reduce_spirv(AtomicFloatOp::Add));
        let min_caps = capabilities(&atomic_float_reduce_spirv(AtomicFloatOp::Min));
        assert!(add_caps.contains(&super::CAPABILITY_ATOMIC_FLOAT32_ADD_EXT));
        assert!(min_caps.contains(&super::CAPABILITY_ATOMIC_FLOAT32_MIN_MAX_EXT));
        assert!(!min_caps.contains(&super::CAPABILITY_ATOMIC_FLOAT32_ADD_EXT));
    }
}
