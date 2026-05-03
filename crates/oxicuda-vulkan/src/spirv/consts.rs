//! SPIR-V constant definitions: opcodes, capabilities, decorations, etc.
//!
//! These constants are shared by all submodules of [`crate::spirv`] and the
//! sibling [`crate::spirv::subgroup`] module.  Visibility is `pub(super)` so
//! every file inside `spirv/` can use them without re-export gymnastics.

// ─── Public version constants ───────────────────────────────

/// SPIR-V magic number.
pub const SPIRV_MAGIC: u32 = 0x07230203;
/// SPIR-V version 1.2.
pub const SPIRV_VERSION_1_2: u32 = 0x0001_0200;
/// SPIR-V version 1.3 (used for compute shaders with StorageBuffer).
pub const SPIRV_VERSION_1_3: u32 = 0x0001_0300;
/// Generator magic — OxiCUDA Vulkan backend.
pub const SPIRV_GENERATOR: u32 = 0x000D_0001;

// ─── SPIR-V opcodes ─────────────────────────────────────────

pub(super) const OP_EXT_INST_IMPORT: u32 = 11;
pub(super) const OP_EXT_INST: u32 = 12;
pub(super) const OP_MEMORY_MODEL: u32 = 14;
pub(super) const OP_ENTRY_POINT: u32 = 15;
pub(super) const OP_EXECUTION_MODE: u32 = 16;
pub(super) const OP_CAPABILITY: u32 = 17;
pub(super) const OP_TYPE_VOID: u32 = 19;
pub(super) const OP_TYPE_BOOL: u32 = 20;
pub(super) const OP_TYPE_INT: u32 = 21;
pub(super) const OP_TYPE_FLOAT: u32 = 22;
pub(super) const OP_TYPE_VECTOR: u32 = 23;
pub(super) const OP_TYPE_ARRAY: u32 = 28;
pub(super) const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
pub(super) const OP_TYPE_STRUCT: u32 = 30;
pub(super) const OP_TYPE_POINTER: u32 = 32;
pub(super) const OP_TYPE_FUNCTION: u32 = 33;
pub(super) const OP_CONSTANT: u32 = 43;
pub(super) const OP_FUNCTION: u32 = 54;
pub(super) const OP_FUNCTION_END: u32 = 56;
pub(super) const OP_VARIABLE: u32 = 59;
pub(super) const OP_LOAD: u32 = 61;
pub(super) const OP_STORE: u32 = 62;
pub(super) const OP_ACCESS_CHAIN: u32 = 65;
pub(super) const OP_DECORATE: u32 = 71;
pub(super) const OP_MEMBER_DECORATE: u32 = 72;
pub(super) const OP_BITCAST: u32 = 124;
pub(super) const OP_F_NEGATE: u32 = 127;
pub(super) const OP_I_ADD: u32 = 128;
pub(super) const OP_F_ADD: u32 = 129;
pub(super) const OP_I_SUB: u32 = 130;
pub(super) const OP_F_SUB: u32 = 131;
pub(super) const OP_I_MUL: u32 = 132;
pub(super) const OP_F_MUL: u32 = 133;
pub(super) const OP_U_DIV: u32 = 134;
pub(super) const OP_F_DIV: u32 = 136;
pub(super) const OP_U_MOD: u32 = 137;
pub(super) const OP_LOGICAL_AND: u32 = 167;
pub(super) const OP_U_LESS_THAN: u32 = 176;
pub(super) const OP_CONVERT_U_TO_F: u32 = 112;
pub(super) const OP_LOOP_MERGE: u32 = 246;
pub(super) const OP_SELECTION_MERGE: u32 = 247;
pub(super) const OP_LABEL: u32 = 248;
pub(super) const OP_BRANCH: u32 = 249;
pub(super) const OP_BRANCH_CONDITIONAL: u32 = 250;
pub(super) const OP_RETURN: u32 = 253;
pub(super) const OP_CONTROL_BARRIER: u32 = 224;

// Subgroup / GroupNonUniform opcodes
pub(super) const OP_GROUP_NON_UNIFORM_I_ADD: u32 = 349;
pub(super) const OP_GROUP_NON_UNIFORM_F_ADD: u32 = 350;
pub(super) const OP_GROUP_NON_UNIFORM_F_MIN: u32 = 354;
pub(super) const OP_GROUP_NON_UNIFORM_F_MAX: u32 = 356;
pub(super) const OP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 345;

// Capabilities
pub(super) const CAPABILITY_SHADER: u32 = 1;
pub(super) const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
pub(super) const CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
pub(super) const CAPABILITY_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;

// Group operation constants (used as operand to GroupNonUniform* instructions)
pub(super) const GROUP_OPERATION_REDUCE: u32 = 0;
pub(super) const GROUP_OPERATION_INCLUSIVE_SCAN: u32 = 1;
// Scope constants
pub(super) const SCOPE_WORKGROUP: u32 = 2;
pub(super) const SCOPE_SUBGROUP: u32 = 3;

// Memory semantics
pub(super) const MEMORY_SEMANTICS_WORKGROUP_MEMORY: u32 = 0x100; // WorkgroupMemory
pub(super) const MEMORY_SEMANTICS_ACQUIRE_RELEASE: u32 = 0x8; // AcquireRelease

// Addressing / memory model
pub(super) const ADDRESSING_MODEL_LOGICAL: u32 = 0;
pub(super) const MEMORY_MODEL_GLSL450: u32 = 1;

// Execution model / mode
pub(super) const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
pub(super) const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Function control
pub(super) const FUNCTION_CONTROL_NONE: u32 = 0;

// Decorations
pub(super) const DECORATION_BLOCK: u32 = 2;
pub(super) const DECORATION_ARRAY_STRIDE: u32 = 6;
pub(super) const DECORATION_BUILTIN: u32 = 11;
pub(super) const DECORATION_BINDING: u32 = 33;
pub(super) const DECORATION_DESCRIPTOR_SET: u32 = 34;
pub(super) const DECORATION_OFFSET: u32 = 35;

// BuiltIn values
pub(super) const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
pub(super) const BUILTIN_SUBGROUP_SIZE: u32 = 36;
pub(super) const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
pub(super) const BUILTIN_NUM_SUBGROUPS: u32 = 38;
pub(super) const BUILTIN_SUBGROUP_ID: u32 = 40;
pub(super) const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;

// Storage class for Workgroup shared memory
pub(super) const STORAGE_CLASS_WORKGROUP: u32 = 4;

// Storage classes
pub(super) const STORAGE_CLASS_INPUT: u32 = 1;
pub(super) const STORAGE_CLASS_FUNCTION: u32 = 7;
pub(super) const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Selection/loop control
pub(super) const SELECTION_CONTROL_NONE: u32 = 0;
pub(super) const LOOP_CONTROL_NONE: u32 = 0;

// GLSL.std.450 extended instruction numbers
pub(super) const GLSL_F_ABS: u32 = 4;
pub(super) const GLSL_TANH: u32 = 21;
pub(super) const GLSL_EXP: u32 = 27;
pub(super) const GLSL_LOG: u32 = 28;
pub(super) const GLSL_SQRT: u32 = 31;
pub(super) const GLSL_F_MIN: u32 = 39;
pub(super) const GLSL_F_MAX: u32 = 40;

/// Workgroup size for 1-D compute shaders.
pub(super) const WORKGROUP_SIZE: u32 = 256;
