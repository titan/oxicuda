//! Subgroup-size-control (`VK_EXT_subgroup_size_control`) negotiation + emission.
//!
//! Vendor GPUs disagree on the natural subgroup ("warp"/"wavefront") width:
//!
//! | Vendor  | Native subgroup size      |
//! |---------|---------------------------|
//! | NVIDIA  | 32                        |
//! | AMD GCN/CDNA | 64 (Wave64)          |
//! | AMD RDNA     | 32 or 64 (configurable) |
//! | Intel Xe     | 8 / 16 / 32           |
//! | Adreno  | 32 (64/128 on some)       |
//! | Mali    | 16 (8/16 on some)         |
//!
//! `VK_EXT_subgroup_size_control` lets the pipeline pin a *required* subgroup
//! size at creation time (within the device's `[min, max]` range) so a
//! reduction emits the optimal number of shuffle steps. This module provides:
//!
//! - [`SubgroupSizeController`] — a host-side negotiation table that, given the
//!   device's advertised `[min, max]` range and a vendor hint, picks the
//!   subgroup size and the matching `VkPipelineShaderStageRequiredSubgroupSize`
//!   value (returned as plain data so it is fully CPU-testable).
//! - [`subgroup_size_aware_reduce_spirv`] — a reduction shader that bakes the
//!   chosen size into a `SpecId`-decorated specialization constant so the same
//!   SPIR-V can be specialised per device without recompilation.

use super::builder::SpvModule;
use super::consts::{
    BUILTIN_GLOBAL_INVOCATION_ID, CAPABILITY_GROUP_NON_UNIFORM,
    CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC, CAPABILITY_SHADER, DECORATION_ARRAY_STRIDE,
    DECORATION_BINDING, DECORATION_BLOCK, DECORATION_BUILTIN, DECORATION_DESCRIPTOR_SET,
    DECORATION_OFFSET, DECORATION_SPEC_ID, FUNCTION_CONTROL_NONE, GROUP_OPERATION_REDUCE,
    OP_GROUP_NON_UNIFORM_F_ADD, OP_U_LESS_THAN, SCOPE_SUBGROUP, SPIRV_VERSION_1_3,
    STORAGE_CLASS_INPUT, STORAGE_CLASS_STORAGE_BUFFER, WORKGROUP_SIZE,
};

/// GPU vendor hint used to bias subgroup-size selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupVendor {
    /// NVIDIA — native warp size 32.
    Nvidia,
    /// AMD GCN / CDNA — native wavefront size 64.
    AmdWave64,
    /// AMD RDNA — supports both 32 and 64; prefer 32 for compute.
    AmdRdna,
    /// Intel Xe — 8/16/32; prefer 32 (best occupancy for wide reductions).
    Intel,
    /// Qualcomm Adreno — 32.
    Adreno,
    /// ARM Mali — 16.
    Mali,
    /// Unknown vendor — fall back to the device's max within bounds.
    Unknown,
}

impl SubgroupVendor {
    /// The vendor's preferred subgroup size (before clamping to the device
    /// range).
    #[must_use]
    pub fn preferred_size(self) -> u32 {
        match self {
            SubgroupVendor::Nvidia | SubgroupVendor::AmdRdna | SubgroupVendor::Adreno => 32,
            SubgroupVendor::AmdWave64 => 64,
            SubgroupVendor::Intel => 32,
            SubgroupVendor::Mali => 16,
            SubgroupVendor::Unknown => 0, // sentinel: caller uses device max
        }
    }
}

/// Result of subgroup-size negotiation for one pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupSizeChoice {
    /// The required subgroup size to pin at pipeline creation.
    pub required_size: u32,
    /// Whether the device range actually permitted the vendor preference.
    pub honored_preference: bool,
    /// `log2(required_size)` — the number of shuffle steps for a full reduction.
    pub shuffle_steps: u32,
}

/// Host-side subgroup-size negotiator.
///
/// Construct it from the device's advertised `minSubgroupSize` /
/// `maxSubgroupSize` (from `VkPhysicalDeviceSubgroupSizeControlProperties`),
/// then call [`SubgroupSizeController::choose`] with a vendor hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupSizeController {
    min_size: u32,
    max_size: u32,
}

impl SubgroupSizeController {
    /// Create a controller from a device's subgroup-size range.
    ///
    /// Both bounds are clamped to powers of two in `[1, 128]`; `min` must not
    /// exceed `max` (they are swapped if they do, defending against malformed
    /// driver data).
    #[must_use]
    pub fn new(min_size: u32, max_size: u32) -> Self {
        let lo = clamp_pow2(min_size.max(1));
        let hi = clamp_pow2(max_size.max(1));
        let (min_size, max_size) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self { min_size, max_size }
    }

    /// The lower bound of the device subgroup-size range.
    #[must_use]
    pub fn min_size(self) -> u32 {
        self.min_size
    }

    /// The upper bound of the device subgroup-size range.
    #[must_use]
    pub fn max_size(self) -> u32 {
        self.max_size
    }

    /// Negotiate the subgroup size for `vendor`.
    ///
    /// The vendor's preferred size is clamped to `[min, max]`; `Unknown`
    /// vendors take the device maximum (the widest single-instruction
    /// reduction).
    #[must_use]
    pub fn choose(self, vendor: SubgroupVendor) -> SubgroupSizeChoice {
        let pref = vendor.preferred_size();
        let required_size = if pref == 0 {
            self.max_size
        } else {
            pref.clamp(self.min_size, self.max_size)
        };
        SubgroupSizeChoice {
            required_size,
            honored_preference: pref != 0 && required_size == pref,
            shuffle_steps: required_size.trailing_zeros(),
        }
    }
}

/// Clamp a value down to the nearest power of two within `[1, 128]`.
fn clamp_pow2(v: u32) -> u32 {
    let v = v.clamp(1, 128);
    if v.is_power_of_two() {
        v
    } else {
        // Largest power of two <= v.
        1u32 << (31 - v.leading_zeros())
    }
}

/// Generate a subgroup reduction shader whose subgroup size is a
/// `SpecId`-decorated specialization constant.
///
/// The host overrides spec-constant id `0` with the negotiated size via
/// `VkSpecializationInfo` at pipeline creation. The constant is referenced in
/// the module (it conditions the bounds check) so it cannot be optimised away.
///
/// Bindings: 0 = input `float[]`, 1 = output `float[]`, 2 = params `uint[]`
/// with `params[0] = count`.
#[must_use]
pub fn subgroup_size_aware_reduce_spirv(default_size: u32) -> Vec<u32> {
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
    let c_scope_subgroup = m.alloc_id();
    let spec_subgroup_size = m.alloc_id();

    let var_gid = m.alloc_id();
    let var_input = m.alloc_id();
    let var_output = m.alloc_id();
    let var_params = m.alloc_id();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM);
    m.emit_capability(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC);

    m.emit_memory_model();
    m.emit_entry_point(main_fn, "main", &[var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);
    // The subgroup size is a specialization constant (SpecId = 0).
    m.emit_decorate(spec_subgroup_size, DECORATION_SPEC_ID, &[0]);
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
    m.emit_constant_u32(ty_uint, c_scope_subgroup, SCOPE_SUBGROUP);
    // Specialization constant for subgroup size (overridable by the host).
    m.emit_spec_constant_u32(ty_uint, spec_subgroup_size, default_size);

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

    // value = input[gid]
    let inp_ptr = m.alloc_id();
    let value = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, inp_ptr, var_input, &[c_uint_0, gid]);
    m.emit_load(ty_float, value, inp_ptr);

    // partial = subgroupAdd(value) over the (spec-sized) subgroup.
    let partial = m.alloc_id();
    m.emit(
        OP_GROUP_NON_UNIFORM_F_ADD,
        &[
            ty_float,
            partial,
            c_scope_subgroup,
            GROUP_OPERATION_REDUCE,
            value,
        ],
    );

    // Use the spec constant: clamp the write index to (gid / subgroup_size).
    // This reference forces the constant to remain live in the module.
    let lane = m.alloc_id();
    m.emit(
        super::consts::OP_U_DIV,
        &[ty_uint, lane, gid, spec_subgroup_size],
    );
    let out_ptr = m.alloc_id();
    m.emit_access_chain(ty_ptr_sb_float, out_ptr, var_output, &[c_uint_0, lane]);
    m.emit_store(out_ptr, partial);

    m.emit_branch(label_merge);
    m.emit_label(label_merge);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::consts::{DECORATION_SPEC_ID, OP_SPEC_CONSTANT, SPIRV_MAGIC};

    #[test]
    fn nvidia_prefers_32_within_range() {
        let c = SubgroupSizeController::new(32, 32);
        let choice = c.choose(SubgroupVendor::Nvidia);
        assert_eq!(choice.required_size, 32);
        assert!(choice.honored_preference);
        assert_eq!(choice.shuffle_steps, 5); // log2(32)
    }

    #[test]
    fn amd_wave64_picks_64_when_allowed() {
        let c = SubgroupSizeController::new(32, 64);
        let choice = c.choose(SubgroupVendor::AmdWave64);
        assert_eq!(choice.required_size, 64);
        assert!(choice.honored_preference);
        assert_eq!(choice.shuffle_steps, 6);
    }

    #[test]
    fn preference_clamped_to_device_max() {
        // Device only supports up to 32 but AMD wants 64 → clamp to 32.
        let c = SubgroupSizeController::new(16, 32);
        let choice = c.choose(SubgroupVendor::AmdWave64);
        assert_eq!(choice.required_size, 32);
        assert!(!choice.honored_preference, "64 was not available");
    }

    #[test]
    fn preference_clamped_to_device_min() {
        // Mali wants 16 but device floor is 32.
        let c = SubgroupSizeController::new(32, 64);
        let choice = c.choose(SubgroupVendor::Mali);
        assert_eq!(choice.required_size, 32);
        assert!(!choice.honored_preference);
    }

    #[test]
    fn unknown_vendor_takes_device_max() {
        let c = SubgroupSizeController::new(8, 32);
        let choice = c.choose(SubgroupVendor::Unknown);
        assert_eq!(choice.required_size, 32);
        assert!(!choice.honored_preference);
    }

    #[test]
    fn intel_8_16_32_negotiation() {
        let c = SubgroupSizeController::new(8, 32);
        assert_eq!(c.choose(SubgroupVendor::Intel).required_size, 32);
        // Narrow device: only 8-wide subgroups.
        let narrow = SubgroupSizeController::new(8, 8);
        assert_eq!(narrow.choose(SubgroupVendor::Intel).required_size, 8);
        assert_eq!(narrow.choose(SubgroupVendor::Intel).shuffle_steps, 3);
    }

    #[test]
    fn malformed_range_is_repaired() {
        // min > max is swapped; non-power-of-two clamped down.
        let c = SubgroupSizeController::new(48, 12);
        assert!(c.min_size() <= c.max_size());
        assert!(c.min_size().is_power_of_two());
        assert!(c.max_size().is_power_of_two());
    }

    #[test]
    fn shader_declares_spec_constant_with_specid_0() {
        let w = subgroup_size_aware_reduce_spirv(32);
        assert_eq!(w[0], SPIRV_MAGIC);

        // Find a SpecId(1) decoration with operand 0, and an OpSpecConstant (50).
        let mut has_spec_id = false;
        let mut has_spec_const = false;
        let mut i = 5usize;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            let opcode = w[i] & 0xFFFF;
            // OpDecorate (71): target, decoration, operand...
            if opcode == 71 && w[i + 2] == DECORATION_SPEC_ID && w[i + 3] == 0 {
                has_spec_id = true;
            }
            if opcode == OP_SPEC_CONSTANT {
                has_spec_const = true;
            }
            i += count;
        }
        assert!(has_spec_id, "missing SpecId decoration");
        assert!(has_spec_const, "missing OpSpecConstant");
    }

    #[test]
    fn default_size_is_baked_into_spec_constant() {
        let w = subgroup_size_aware_reduce_spirv(64);
        let mut found = false;
        let mut i = 5usize;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            // OpSpecConstant (50): result-type, result-id, value.
            if (w[i] & 0xFFFF) == OP_SPEC_CONSTANT && w[i + 3] == 64 {
                found = true;
            }
            i += count;
        }
        assert!(found, "default subgroup size 64 should be baked in");
    }
}
