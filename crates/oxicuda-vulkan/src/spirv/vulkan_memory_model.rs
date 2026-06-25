//! Vulkan-memory-model explicit availability/visibility SPIR-V generator.
//!
//! Vulkan 1.2 promotes the *Vulkan memory model* to core. Instead of relying
//! on coarse `vkCmdPipelineBarrier` calls between dispatches, shaders can
//! annotate individual `OpLoad`/`OpStore` instructions with the
//! `MakePointerAvailable` / `MakePointerVisible` memory operands plus a
//! synchronisation scope, expressing exactly which writes must be made
//! available to which readers.
//!
//! This generator emits a buffer-copy shader (`output[i] = input[i]`) where the
//! store is tagged *available* to the `QueueFamily` scope and a subsequent load
//! is tagged *visible*, demonstrating the explicit acquire/release semantics
//! that replace a global barrier. The [`VulkanMemModel`] helper records the
//! scope/operand configuration as a host-side data structure so the choice can
//! be inspected and unit-tested without a device.

use super::builder::SpvModule;
use super::consts::{
    BUILTIN_GLOBAL_INVOCATION_ID, CAPABILITY_SHADER, CAPABILITY_VULKAN_MEMORY_MODEL,
    DECORATION_ARRAY_STRIDE, DECORATION_BINDING, DECORATION_BLOCK, DECORATION_BUILTIN,
    DECORATION_DESCRIPTOR_SET, DECORATION_OFFSET, FUNCTION_CONTROL_NONE,
    MEMORY_OPERAND_MAKE_POINTER_AVAILABLE, MEMORY_OPERAND_MAKE_POINTER_VISIBLE,
    MEMORY_OPERAND_NON_PRIVATE_POINTER, OP_LOAD, OP_STORE, OP_U_LESS_THAN, SCOPE_DEVICE,
    SCOPE_QUEUE_FAMILY, SCOPE_WORKGROUP, SPIRV_VERSION_1_5, STORAGE_CLASS_INPUT,
    STORAGE_CLASS_STORAGE_BUFFER, WORKGROUP_SIZE,
};

/// Synchronisation scope for an availability/visibility operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemScope {
    /// Visible to the whole device.
    Device,
    /// Visible within a single queue family (cross-dispatch on one queue).
    QueueFamily,
    /// Visible within a workgroup.
    Workgroup,
}

impl MemScope {
    /// SPIR-V scope id for this scope.
    #[must_use]
    pub fn spirv_scope(self) -> u32 {
        match self {
            MemScope::Device => SCOPE_DEVICE,
            MemScope::QueueFamily => SCOPE_QUEUE_FAMILY,
            MemScope::Workgroup => SCOPE_WORKGROUP,
        }
    }
}

/// Host-side description of the Vulkan-memory-model annotations a shader uses.
///
/// Captures the scope at which writes become available and reads become
/// visible. The same configuration can be fed to a `vkCmdPipelineBarrier`
/// emitted with `VK_KHR_synchronization2` so the barrier and the in-shader
/// operands agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanMemModel {
    /// Scope at which a store becomes available.
    pub available_scope: MemScope,
    /// Scope at which a load becomes visible.
    pub visible_scope: MemScope,
    /// Whether the buffer is `NonPrivate` (shared between invocations).
    pub non_private: bool,
}

impl Default for VulkanMemModel {
    fn default() -> Self {
        Self {
            available_scope: MemScope::QueueFamily,
            visible_scope: MemScope::QueueFamily,
            non_private: true,
        }
    }
}

impl VulkanMemModel {
    /// Create a configuration that makes writes available and reads visible at
    /// device scope (the strongest, cross-queue guarantee).
    #[must_use]
    pub fn device_scope() -> Self {
        Self {
            available_scope: MemScope::Device,
            visible_scope: MemScope::Device,
            non_private: true,
        }
    }

    /// Memory-operand bitmask for an availability store (operand 4 of `OpStore`).
    #[must_use]
    pub fn store_operands(self) -> u32 {
        let mut bits = MEMORY_OPERAND_MAKE_POINTER_AVAILABLE;
        if self.non_private {
            bits |= MEMORY_OPERAND_NON_PRIVATE_POINTER;
        }
        bits
    }

    /// Memory-operand bitmask for a visibility load (operand 3 of `OpLoad`).
    #[must_use]
    pub fn load_operands(self) -> u32 {
        let mut bits = MEMORY_OPERAND_MAKE_POINTER_VISIBLE;
        if self.non_private {
            bits |= MEMORY_OPERAND_NON_PRIVATE_POINTER;
        }
        bits
    }
}

/// Generate a buffer-copy shader using explicit Vulkan-memory-model operands.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`, 2 = params `uint[]`
/// with `params[0] = count`. The load is tagged visible and the store is
/// tagged available at the scopes recorded in `model`.
#[must_use]
pub fn vulkan_memory_model_copy_spirv(model: VulkanMemModel) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_5);

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
    let c_scope_avail = m.alloc_id();
    let c_scope_visible = m.alloc_id();

    let var_gid = m.alloc_id();
    let var_input = m.alloc_id();
    let var_output = m.alloc_id();
    let var_params = m.alloc_id();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(CAPABILITY_VULKAN_MEMORY_MODEL);

    // Vulkan memory model (id 3) — required when using the make-available/
    // make-visible operands.
    m.emit(
        super::consts::OP_MEMORY_MODEL,
        &[super::consts::ADDRESSING_MODEL_LOGICAL, 3],
    );
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

    // ── Constants: scope ids carried as memory-operand id operands ──
    m.emit_constant_u32(ty_uint, c_uint_0, 0);
    m.emit_constant_u32(ty_uint, c_scope_avail, model.available_scope.spirv_scope());
    m.emit_constant_u32(ty_uint, c_scope_visible, model.visible_scope.spirv_scope());

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

    // ── Function ──
    let label_entry = m.alloc_id();
    let label_body = m.alloc_id();
    let label_merge = m.alloc_id();

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn_void);
    m.emit_label(label_entry);

    let gid_ptr = m.alloc_id();
    let gid = m.alloc_id();
    m.emit_access_chain(ty_ptr_input_uint, gid_ptr, var_gid, &[c_uint_0]);
    m.emit_load(ty_uint, gid, gid_ptr);

    let cnt_ptr = m.alloc_id();
    let count = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_uint, cnt_ptr, var_params, &[c_uint_0, c_uint_0]);
    m.emit_load(ty_uint, count, cnt_ptr);

    let cond = m.alloc_id();
    m.emit(OP_U_LESS_THAN, &[ty_bool, cond, gid, count]);
    m.emit_selection_merge(label_merge);
    m.emit_branch_conditional(cond, label_body, label_merge);

    m.emit_label(label_body);

    // value = OpLoad input[gid] with MakePointerVisible|NonPrivate, scope id.
    let inp_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, inp_ptr, var_input, &[c_uint_0, gid]);
    let value = m.alloc_id();
    m.emit(
        OP_LOAD,
        &[
            ty_float,
            value,
            inp_ptr,
            model.load_operands(),
            c_scope_visible,
        ],
    );

    // OpStore output[gid] = value with MakePointerAvailable|NonPrivate, scope id.
    let out_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, out_ptr, var_output, &[c_uint_0, gid]);
    m.emit(
        OP_STORE,
        &[out_ptr, value, model.store_operands(), c_scope_avail],
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
        CAPABILITY_VULKAN_MEMORY_MODEL, MEMORY_OPERAND_MAKE_POINTER_AVAILABLE,
        MEMORY_OPERAND_MAKE_POINTER_VISIBLE, MEMORY_OPERAND_NON_PRIVATE_POINTER, SCOPE_DEVICE,
        SCOPE_QUEUE_FAMILY, SPIRV_MAGIC, SPIRV_VERSION_1_5,
    };

    #[test]
    fn header_is_spirv_15() {
        let w = vulkan_memory_model_copy_spirv(VulkanMemModel::default());
        assert_eq!(w[0], SPIRV_MAGIC);
        assert_eq!(w[1], SPIRV_VERSION_1_5);
        assert!(w[3] > 0);
        assert_eq!(w[4], 0);
    }

    #[test]
    fn emits_vulkan_memory_model_capability_and_model_id() {
        let w = vulkan_memory_model_copy_spirv(VulkanMemModel::default());
        let mut has_cap = false;
        let mut model_id = None;
        let mut i = 5usize;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            let opcode = w[i] & 0xFFFF;
            if opcode == 17 && w[i + 1] == CAPABILITY_VULKAN_MEMORY_MODEL {
                has_cap = true;
            }
            if opcode == 14 {
                // OpMemoryModel addressing, memory-model
                model_id = Some(w[i + 2]);
            }
            i += count;
        }
        assert!(has_cap, "missing VulkanMemoryModel capability");
        assert_eq!(model_id, Some(3), "memory model must be Vulkan (3)");
    }

    #[test]
    fn operand_masks_set_correct_bits() {
        let m = VulkanMemModel::default();
        assert_eq!(
            m.store_operands(),
            MEMORY_OPERAND_MAKE_POINTER_AVAILABLE | MEMORY_OPERAND_NON_PRIVATE_POINTER
        );
        assert_eq!(
            m.load_operands(),
            MEMORY_OPERAND_MAKE_POINTER_VISIBLE | MEMORY_OPERAND_NON_PRIVATE_POINTER
        );

        let private = VulkanMemModel {
            non_private: false,
            ..VulkanMemModel::default()
        };
        assert_eq!(
            private.store_operands(),
            MEMORY_OPERAND_MAKE_POINTER_AVAILABLE
        );
        assert_eq!(private.load_operands(), MEMORY_OPERAND_MAKE_POINTER_VISIBLE);
    }

    #[test]
    fn scope_selection_maps_to_spirv_ids() {
        assert_eq!(MemScope::Device.spirv_scope(), SCOPE_DEVICE);
        assert_eq!(MemScope::QueueFamily.spirv_scope(), SCOPE_QUEUE_FAMILY);
        assert_eq!(
            VulkanMemModel::device_scope().available_scope,
            MemScope::Device
        );
    }

    #[test]
    fn device_scope_constant_appears_in_module() {
        let w = vulkan_memory_model_copy_spirv(VulkanMemModel::device_scope());
        // Look for OpConstant (43) emitting the Device scope value.
        let mut found = false;
        let mut i = 5usize;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            if (w[i] & 0xFFFF) == 43 && w[i + 3] == SCOPE_DEVICE {
                found = true;
            }
            i += count;
        }
        assert!(found, "Device scope constant should be emitted");
    }
}
