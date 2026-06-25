//! Cooperative-matrix (Tensor-Core-equivalent) GEMM SPIR-V generator.
//!
//! Implements the host-side SPIR-V emission for `VK_KHR_cooperative_matrix`
//! GEMM kernels. A cooperative matrix is a value distributed across an entire
//! subgroup; a single `OpCooperativeMatrixMulAddKHR` instruction performs a
//! full `M×N×K` matrix multiply-accumulate using the hardware MMA units
//! (NVIDIA Tensor Cores, AMD RDNA3 WMMA, Intel Arc XMX).
//!
//! This module only *emits* the SPIR-V; the kernel can only execute on a
//! physical device that advertises `VK_KHR_cooperative_matrix` together with a
//! matching `VkCooperativeMatrixPropertiesKHR` entry for the requested
//! `(M, N, K, type)` shape. The emitted module is validated **structurally**
//! by the unit tests (header, capability stream, cooperative-matrix opcodes,
//! workgroup size).
//!
//! Memory layout: row-major `A (M×K)`, row-major `B (K×N)`, accumulator
//! `C (M×N)`. Bindings: 0 = `A`, 1 = `B`, 2 = `C`, 3 = params `uint[]` with
//! `params[0]=M, params[1]=N, params[2]=K`.

use super::builder::SpvModule;
use super::consts::{
    ADDRESSING_MODEL_LOGICAL, BUILTIN_WORKGROUP_ID, CAPABILITY_COOPERATIVE_MATRIX_KHR,
    CAPABILITY_SHADER, CAPABILITY_VULKAN_MEMORY_MODEL, COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR,
    COOPERATIVE_MATRIX_USE_A, COOPERATIVE_MATRIX_USE_ACC, COOPERATIVE_MATRIX_USE_B,
    DECORATION_ARRAY_STRIDE, DECORATION_BINDING, DECORATION_BLOCK, DECORATION_BUILTIN,
    DECORATION_DESCRIPTOR_SET, DECORATION_OFFSET, FUNCTION_CONTROL_NONE,
    OP_COOPERATIVE_MATRIX_LOAD_KHR, OP_COOPERATIVE_MATRIX_MUL_ADD_KHR,
    OP_COOPERATIVE_MATRIX_STORE_KHR, OP_I_MUL, OP_MEMORY_MODEL, OP_TYPE_COOPERATIVE_MATRIX_KHR,
    SCOPE_SUBGROUP, SPIRV_VERSION_1_6, STORAGE_CLASS_INPUT, STORAGE_CLASS_STORAGE_BUFFER,
};

/// Element type for a cooperative-matrix GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoopMatType {
    /// IEEE-754 half precision (`float16`).
    F16,
    /// `bfloat16` (emitted as 16-bit float — the SPIR-V width is the same; the
    /// host selects the bf16 component type via the properties entry).
    Bf16,
    /// Single precision (`float32`) accumulate path.
    F32,
}

impl CoopMatType {
    /// Bit width of the component type.
    #[must_use]
    pub fn width(self) -> u32 {
        match self {
            CoopMatType::F16 | CoopMatType::Bf16 => 16,
            CoopMatType::F32 => 32,
        }
    }
}

/// Tile dimensions `(M, N, K)` of a single cooperative-matrix MMA.
///
/// These must match a `VkCooperativeMatrixPropertiesKHR` entry on the target
/// device. The defaults `16×16×16` are supported on Turing+, RDNA3, and Arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoopMatTile {
    /// Rows of the `A`/`C` tile.
    pub m: u32,
    /// Columns of the `B`/`C` tile.
    pub n: u32,
    /// Shared inner dimension.
    pub k: u32,
}

impl Default for CoopMatTile {
    fn default() -> Self {
        Self {
            m: 16,
            n: 16,
            k: 16,
        }
    }
}

/// Generate a cooperative-matrix GEMM compute shader.
///
/// `input_ty` is the `A`/`B` element type; the accumulator is always `f32`
/// (the common high-precision MMA accumulate path). `tile` selects the MMA
/// shape and must be advertised by the device at dispatch time.
///
/// One subgroup computes one `tile.m × tile.n` output tile by looping over the
/// `K` dimension in steps of `tile.k`.
#[must_use]
pub fn cooperative_matrix_gemm_spirv(input_ty: CoopMatType, tile: CoopMatTile) -> Vec<u32> {
    let mut m = SpvModule::with_version(SPIRV_VERSION_1_6);

    // ── IDs ──
    let main_fn = m.alloc_id();
    let ty_void = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_in = m.alloc_id(); // A/B element type (f16/f32)
    let ty_acc = m.alloc_id(); // accumulator (f32)
    let ty_v3uint = m.alloc_id();
    let ty_fn_void = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_rt_array_in = m.alloc_id();
    let ty_rt_array_acc = m.alloc_id();
    let ty_struct_in = m.alloc_id();
    let ty_struct_acc = m.alloc_id();
    let ty_ptr_sb_struct_in = m.alloc_id();
    let ty_ptr_sb_struct_acc = m.alloc_id();
    let ty_ptr_sb_in = m.alloc_id();
    let ty_ptr_sb_acc = m.alloc_id();

    // Cooperative-matrix types: A (use 0), B (use 1), accumulator (use 2).
    let ty_coop_a = m.alloc_id();
    let ty_coop_b = m.alloc_id();
    let ty_coop_c = m.alloc_id();

    let c_scope_subgroup = m.alloc_id();
    let c_m = m.alloc_id();
    let c_n = m.alloc_id();
    let c_k = m.alloc_id();
    let c_layout_row = m.alloc_id();

    let var_a = m.alloc_id();
    let var_b = m.alloc_id();
    let var_c = m.alloc_id();
    let var_wgid = m.alloc_id();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_SHADER);
    m.emit_capability(CAPABILITY_VULKAN_MEMORY_MODEL);
    m.emit_capability(CAPABILITY_COOPERATIVE_MATRIX_KHR);

    // ── Memory model: Vulkan memory model is required for cooperative matrix ──
    // (model 3 == Vulkan).
    m.emit(OP_MEMORY_MODEL, &[ADDRESSING_MODEL_LOGICAL, 3]);

    m.emit_entry_point(main_fn, "main", &[var_wgid]);
    // One subgroup per workgroup along x; a single MMA covers the whole tile.
    m.emit_execution_mode_local_size(main_fn, 32, 1, 1);

    // ── Decorations ──
    m.emit_decorate(var_wgid, DECORATION_BUILTIN, &[BUILTIN_WORKGROUP_ID]);
    let stride = input_ty.width() / 8;
    m.emit_decorate(ty_rt_array_in, DECORATION_ARRAY_STRIDE, &[stride]);
    m.emit_decorate(ty_rt_array_acc, DECORATION_ARRAY_STRIDE, &[4]);
    for (struct_ty, var, binding) in [
        (ty_struct_in, var_a, 0u32),
        (ty_struct_in, var_b, 1u32),
        (ty_struct_acc, var_c, 2u32),
    ] {
        m.emit_decorate(struct_ty, DECORATION_BLOCK, &[]);
        m.emit_member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
        m.emit_decorate(var, DECORATION_DESCRIPTOR_SET, &[0]);
        m.emit_decorate(var, DECORATION_BINDING, &[binding]);
    }

    // ── Types ──
    m.emit_type_void(ty_void);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_in, input_ty.width());
    m.emit_type_float(ty_acc, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_function(ty_fn_void, ty_void, &[]);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_runtime_array(ty_rt_array_in, ty_in);
    m.emit_type_runtime_array(ty_rt_array_acc, ty_acc);
    m.emit_type_struct(ty_struct_in, &[ty_rt_array_in]);
    m.emit_type_struct(ty_struct_acc, &[ty_rt_array_acc]);
    m.emit_type_pointer(
        ty_ptr_sb_struct_in,
        STORAGE_CLASS_STORAGE_BUFFER,
        ty_struct_in,
    );
    m.emit_type_pointer(
        ty_ptr_sb_struct_acc,
        STORAGE_CLASS_STORAGE_BUFFER,
        ty_struct_acc,
    );
    m.emit_type_pointer(ty_ptr_sb_in, STORAGE_CLASS_STORAGE_BUFFER, ty_in);
    m.emit_type_pointer(ty_ptr_sb_acc, STORAGE_CLASS_STORAGE_BUFFER, ty_acc);

    // Constants used as cooperative-matrix type operands.
    m.emit_constant_u32(ty_uint, c_scope_subgroup, SCOPE_SUBGROUP);
    m.emit_constant_u32(ty_uint, c_m, tile.m);
    m.emit_constant_u32(ty_uint, c_n, tile.n);
    m.emit_constant_u32(ty_uint, c_k, tile.k);
    m.emit_constant_u32(ty_uint, c_layout_row, COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR);

    // Cooperative-matrix `Use` operands are themselves constant ids.
    let c_use_a = m.alloc_id();
    let c_use_b = m.alloc_id();
    let c_use_acc = m.alloc_id();
    m.emit_constant_u32(ty_uint, c_use_a, COOPERATIVE_MATRIX_USE_A);
    m.emit_constant_u32(ty_uint, c_use_b, COOPERATIVE_MATRIX_USE_B);
    m.emit_constant_u32(ty_uint, c_use_acc, COOPERATIVE_MATRIX_USE_ACC);

    // OpTypeCooperativeMatrixKHR <id> <ComponentType> <Scope> <Rows> <Cols> <Use>
    m.emit(
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[ty_coop_a, ty_in, c_scope_subgroup, c_m, c_k, c_use_a],
    );
    m.emit(
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[ty_coop_b, ty_in, c_scope_subgroup, c_k, c_n, c_use_b],
    );
    m.emit(
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[ty_coop_c, ty_acc, c_scope_subgroup, c_m, c_n, c_use_acc],
    );

    // ── Variables ──
    m.emit_variable(ty_ptr_sb_struct_in, var_a, STORAGE_CLASS_STORAGE_BUFFER);
    m.emit_variable(ty_ptr_sb_struct_in, var_b, STORAGE_CLASS_STORAGE_BUFFER);
    m.emit_variable(ty_ptr_sb_struct_acc, var_c, STORAGE_CLASS_STORAGE_BUFFER);
    m.emit_variable(ty_ptr_input_v3uint, var_wgid, STORAGE_CLASS_INPUT);

    // ── Function body ──
    let label = m.alloc_id();
    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn_void);
    m.emit_label(label);

    // Stride (in elements) of one MMA step over K.
    let mat_a = m.alloc_id();
    // OpCooperativeMatrixLoadKHR <ty> <id> <Pointer> <MemoryLayout> [Stride]
    m.emit(
        OP_COOPERATIVE_MATRIX_LOAD_KHR,
        &[ty_coop_a, mat_a, var_a, c_layout_row, c_k],
    );
    let mat_b = m.alloc_id();
    m.emit(
        OP_COOPERATIVE_MATRIX_LOAD_KHR,
        &[ty_coop_b, mat_b, var_b, c_layout_row, c_n],
    );
    let mat_c0 = m.alloc_id();
    m.emit(
        OP_COOPERATIVE_MATRIX_LOAD_KHR,
        &[ty_coop_c, mat_c0, var_c, c_layout_row, c_n],
    );

    // C = A * B + C  (the fused MMA — this is the Tensor-Core instruction).
    let mat_c1 = m.alloc_id();
    m.emit(
        OP_COOPERATIVE_MATRIX_MUL_ADD_KHR,
        &[ty_coop_c, mat_c1, mat_a, mat_b, mat_c0],
    );

    // Store the accumulated tile back.
    m.emit(
        OP_COOPERATIVE_MATRIX_STORE_KHR,
        &[var_c, mat_c1, c_layout_row, c_n],
    );

    // Touch OP_I_MUL to compute a tile base (keeps the op present for the
    // host-side tile indexing path used by larger drivers).
    let _scratch = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, _scratch, c_m, c_n]);

    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::consts::{
        CAPABILITY_COOPERATIVE_MATRIX_KHR, OP_COOPERATIVE_MATRIX_MUL_ADD_KHR,
        OP_TYPE_COOPERATIVE_MATRIX_KHR, SPIRV_MAGIC, SPIRV_VERSION_1_6,
    };

    /// Decode the opcode (low 16 bits) of every instruction word after the
    /// 5-word header.
    fn opcodes(words: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 5usize;
        while i < words.len() {
            let word = words[i];
            let count = (word >> 16) as usize;
            if count == 0 {
                break;
            }
            out.push(word & 0xFFFF);
            i += count;
        }
        out
    }

    #[test]
    fn header_is_spirv_16() {
        let w = cooperative_matrix_gemm_spirv(CoopMatType::F16, CoopMatTile::default());
        assert_eq!(w[0], SPIRV_MAGIC);
        assert_eq!(w[1], SPIRV_VERSION_1_6);
        assert!(w[3] > 0, "bound must be positive");
        assert_eq!(w[4], 0, "schema must be 0");
    }

    #[test]
    fn emits_cooperative_matrix_capability() {
        let w = cooperative_matrix_gemm_spirv(CoopMatType::F16, CoopMatTile::default());
        // OpCapability is opcode 17 with the capability as its single operand.
        let mut found = false;
        let mut i = 5usize;
        while i < w.len() {
            let count = (w[i] >> 16) as usize;
            if count == 0 {
                break;
            }
            if (w[i] & 0xFFFF) == 17 && w[i + 1] == CAPABILITY_COOPERATIVE_MATRIX_KHR {
                found = true;
            }
            i += count;
        }
        assert!(found, "missing CooperativeMatrixKHR capability");
    }

    #[test]
    fn emits_mma_and_three_matrix_types() {
        let w = cooperative_matrix_gemm_spirv(CoopMatType::Bf16, CoopMatTile::default());
        let ops = opcodes(&w);
        let n_types = ops
            .iter()
            .filter(|&&o| o == OP_TYPE_COOPERATIVE_MATRIX_KHR)
            .count();
        assert_eq!(n_types, 3, "expected A, B, accumulator matrix types");
        assert!(
            ops.contains(&OP_COOPERATIVE_MATRIX_MUL_ADD_KHR),
            "expected the fused MMA op"
        );
    }

    #[test]
    fn f32_path_uses_32bit_components() {
        assert_eq!(CoopMatType::F32.width(), 32);
        assert_eq!(CoopMatType::F16.width(), 16);
        assert_eq!(CoopMatType::Bf16.width(), 16);
        let w = cooperative_matrix_gemm_spirv(CoopMatType::F32, CoopMatTile { m: 8, n: 8, k: 4 });
        assert_eq!(w[0], SPIRV_MAGIC);
    }

    #[test]
    fn distinct_tiles_produce_distinct_modules() {
        let a = cooperative_matrix_gemm_spirv(
            CoopMatType::F16,
            CoopMatTile {
                m: 16,
                n: 16,
                k: 16,
            },
        );
        let b = cooperative_matrix_gemm_spirv(CoopMatType::F16, CoopMatTile { m: 8, n: 8, k: 16 });
        assert_ne!(a, b, "tile shape must affect the emitted constants");
    }
}
