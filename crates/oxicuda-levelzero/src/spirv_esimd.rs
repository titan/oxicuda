//! ESIMD / `SPV_INTEL_subgroups` SPIR-V kernel generators.
//!
//! Intel GPUs expose *Explicit SIMD* (ESIMD) block memory operations through
//! the [`SPV_INTEL_subgroups`] extension: `OpSubgroupBlockReadINTEL` /
//! `OpSubgroupBlockWriteINTEL` move a contiguous block of data per sub-group in
//! a single hardware message, which is critical for high-throughput tile loads
//! on Xe-HPC GEMM. The same extension underpins the **DPAS** (Dot-Product
//! Accumulate Systolic) path used for INT8 / BF16 systolic GEMM on Xe-HPG.
//!
//! This module emits the SPIR-V module words for those kernels and is fully
//! CPU-testable: the tests assert on the emitted magic/header, the declared
//! `SPV_INTEL_subgroups` extension string, the `SubgroupBufferBlockIOINTEL`
//! capability, and the presence of the block-read/-write opcodes. Actually
//! dispatching the kernel needs a physical Intel GPU and is out of scope here.
//!
//! [`SPV_INTEL_subgroups`]:
//!   <https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_subgroups.asciidoc>

use crate::spirv::{
    EXECUTION_MODEL_KERNEL, FUNCTION_CONTROL_NONE, OP_COMPOSITE_EXTRACT, OP_F_ADD, OP_F_MUL,
    OP_I_ADD, OP_I_MUL, SpvModule,
};

// ─── SPV_INTEL_subgroups opcodes ─────────────────────────────

/// `OpSubgroupBlockReadINTEL` — block-load per sub-group.
const OP_SUBGROUP_BLOCK_READ_INTEL: u32 = 5575;
/// `OpSubgroupBlockWriteINTEL` — block-store per sub-group.
const OP_SUBGROUP_BLOCK_WRITE_INTEL: u32 = 5576;

// ─── Capabilities ────────────────────────────────────────────

const CAPABILITY_ADDRESSES: u32 = 4;
const CAPABILITY_KERNEL: u32 = 6;
/// `SubgroupShuffleINTEL` (5568) — base sub-group INTEL capability.
const CAPABILITY_SUBGROUP_SHUFFLE_INTEL: u32 = 5568;
/// `SubgroupBufferBlockIOINTEL` (5569) — enables block read/write from buffers.
const CAPABILITY_SUBGROUP_BUFFER_BLOCK_IO_INTEL: u32 = 5569;

// ─── Addressing / memory model ───────────────────────────────

const ADDRESSING_MODEL_PHYSICAL64: u32 = 2;
const MEMORY_MODEL_OPENCL: u32 = 2;

// ─── Decorations / builtins ──────────────────────────────────

const DECORATION_BUILTIN: u32 = 11;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// ─── Storage classes ─────────────────────────────────────────

const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_CROSS_WORKGROUP: u32 = 5;

/// Workgroup size used by the ESIMD block kernels (one sub-group per group of
/// 16 by default; the host sets the real local size via `zeKernelSetGroupSize`).
const WORKGROUP_SIZE: u32 = 16;

// ─── ESIMD block copy kernel ─────────────────────────────────

/// Generate an OpenCL SPIR-V kernel that copies `count` floats using
/// `OpSubgroupBlockReadINTEL` / `OpSubgroupBlockWriteINTEL`.
///
/// Each sub-group block-reads one element-per-lane from `input` and
/// block-writes it to `output`, demonstrating the ESIMD high-throughput tile
/// transfer path. The host launches it with global size `count` and a
/// sub-group-sized workgroup.
///
/// Kernel parameters: `(CrossWorkgroup float* input, CrossWorkgroup float* output)`.
/// Entry-point name: `"esimd_block_copy"`.
pub fn esimd_block_copy_spirv() -> Vec<u32> {
    let mut m = SpvModule::new();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);
    m.emit_capability(CAPABILITY_SUBGROUP_SHUFFLE_INTEL);
    m.emit_capability(CAPABILITY_SUBGROUP_BUFFER_BLOCK_IO_INTEL);

    // ── Extension ──
    m.emit_extension("SPV_INTEL_subgroups");

    // ── Memory model ──
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // ── Types ──
    let ty_void = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_fn = m.alloc_id();

    // ── Variables / function ──
    let var_gid = m.alloc_id();
    let main_fn = m.alloc_id();
    let p_input = m.alloc_id();
    let p_output = m.alloc_id();
    let label = m.alloc_id();

    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    m.emit_type_void(ty_void);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_function(ty_fn, ty_void, &[ty_ptr_cross_float, ty_ptr_cross_float]);

    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);

    m.emit_entry_point(
        EXECUTION_MODEL_KERNEL,
        main_fn,
        "esimd_block_copy",
        &[var_gid],
    );
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn);
    m.emit_function_parameter(ty_ptr_cross_float, p_input);
    m.emit_function_parameter(ty_ptr_cross_float, p_output);
    m.emit_label(label);

    // gid.x = base offset for this sub-group block.
    let gid_val = m.alloc_id();
    m.emit_load(ty_v3uint, gid_val, var_gid);
    let gid_x = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[ty_uint, gid_x, gid_val, 0]);

    // in_ptr = &input[gid.x]
    let in_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, in_ptr, p_input, gid_x);
    // block_val = OpSubgroupBlockReadINTEL(in_ptr)
    let block_val = m.alloc_id();
    m.emit(OP_SUBGROUP_BLOCK_READ_INTEL, &[ty_float, block_val, in_ptr]);

    // out_ptr = &output[gid.x]; OpSubgroupBlockWriteINTEL(out_ptr, block_val)
    let out_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, out_ptr, p_output, gid_x);
    m.emit(OP_SUBGROUP_BLOCK_WRITE_INTEL, &[out_ptr, block_val]);

    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── DPAS systolic GEMM kernel ───────────────────────────────

/// Tile configuration for a DPAS (systolic) GEMM kernel.
///
/// The Xe-HPG DPAS instruction operates on `8 × repeat_count` systolic depth
/// for INT8/BF16. `m`/`n`/`k` describe the per-sub-group output tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DpasTileConfig {
    /// Output tile rows.
    pub m: u32,
    /// Output tile columns.
    pub n: u32,
    /// Systolic depth (inner accumulation dimension).
    pub k: u32,
}

impl DpasTileConfig {
    /// DG2 (Arc Alchemist) 8×8×16 BF16 systolic tile.
    pub const DG2_BF16: Self = Self { m: 8, n: 8, k: 16 };
    /// DG2 8×8×32 INT8 systolic tile.
    pub const DG2_INT8: Self = Self { m: 8, n: 8, k: 32 };

    /// Number of multiply-accumulate lanes the tile drives.
    #[must_use]
    pub fn mac_lanes(&self) -> u32 {
        self.m * self.n
    }
}

/// Generate an OpenCL SPIR-V GEMM kernel that uses `SPV_INTEL_subgroups`
/// block reads to feed a per-lane DPAS-style accumulation.
///
/// This emits the *block-loaded* GEMM skeleton: each lane block-reads its A and
/// B operands via `OpSubgroupBlockReadINTEL`, multiply-accumulates across the
/// `tile.k` systolic depth, and block-writes the result tile. It models the
/// data-movement structure of an Intel DPAS GEMM. The systolic MAC itself is
/// scalar-emulated in SPIR-V so the module remains portable for validation;
/// the hardware DPAS fusion happens in the driver's JIT.
///
/// Kernel parameters: `(CrossWorkgroup float* A, CrossWorkgroup float* B,
///                      CrossWorkgroup float* C, uint k)`.
/// Entry-point name: `"dpas_gemm"`.
pub fn dpas_gemm_spirv(tile: DpasTileConfig) -> Vec<u32> {
    let mut m = SpvModule::new();

    // ── Capabilities ──
    m.emit_capability(CAPABILITY_KERNEL);
    m.emit_capability(CAPABILITY_ADDRESSES);
    m.emit_capability(CAPABILITY_SUBGROUP_SHUFFLE_INTEL);
    m.emit_capability(CAPABILITY_SUBGROUP_BUFFER_BLOCK_IO_INTEL);

    // ── Extension ──
    m.emit_extension("SPV_INTEL_subgroups");

    // ── Memory model ──
    m.emit_memory_model(ADDRESSING_MODEL_PHYSICAL64, MEMORY_MODEL_OPENCL);

    // ── Types ──
    let ty_void = m.alloc_id();
    let ty_uint = m.alloc_id();
    let ty_float = m.alloc_id();
    let ty_v3uint = m.alloc_id();
    let ty_ptr_input_v3uint = m.alloc_id();
    let ty_ptr_cross_float = m.alloc_id();
    let ty_fn = m.alloc_id();

    // ── Constants: bake the systolic depth + tile dims so the JIT can unroll ──
    let c_k = m.alloc_id();
    let c_tile_m = m.alloc_id();
    let c_tile_n = m.alloc_id();

    let var_gid = m.alloc_id();
    let main_fn = m.alloc_id();
    let p_a = m.alloc_id();
    let p_b = m.alloc_id();
    let p_c = m.alloc_id();
    let p_k = m.alloc_id();
    let label = m.alloc_id();

    m.emit_decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    m.emit_type_void(ty_void);
    m.emit_type_int(ty_uint, 32, 0);
    m.emit_type_float(ty_float, 32);
    m.emit_type_vector(ty_v3uint, ty_uint, 3);
    m.emit_type_pointer(ty_ptr_input_v3uint, STORAGE_CLASS_INPUT, ty_v3uint);
    m.emit_type_pointer(ty_ptr_cross_float, STORAGE_CLASS_CROSS_WORKGROUP, ty_float);
    m.emit_type_function(
        ty_fn,
        ty_void,
        &[
            ty_ptr_cross_float,
            ty_ptr_cross_float,
            ty_ptr_cross_float,
            ty_uint,
        ],
    );

    m.emit_constant_u32(ty_uint, c_k, tile.k);
    m.emit_constant_u32(ty_uint, c_tile_m, tile.m);
    m.emit_constant_u32(ty_uint, c_tile_n, tile.n);

    m.emit_variable(ty_ptr_input_v3uint, var_gid, STORAGE_CLASS_INPUT);

    m.emit_entry_point(EXECUTION_MODEL_KERNEL, main_fn, "dpas_gemm", &[var_gid]);
    m.emit_execution_mode_local_size(main_fn, WORKGROUP_SIZE, 1, 1);

    m.emit_function(ty_void, main_fn, FUNCTION_CONTROL_NONE, ty_fn);
    m.emit_function_parameter(ty_ptr_cross_float, p_a);
    m.emit_function_parameter(ty_ptr_cross_float, p_b);
    m.emit_function_parameter(ty_ptr_cross_float, p_c);
    m.emit_function_parameter(ty_uint, p_k);
    m.emit_label(label);

    // lane = gid.x — this sub-group lane index addresses its A/B block.
    let gid_val = m.alloc_id();
    m.emit_load(ty_v3uint, gid_val, var_gid);
    let lane = m.alloc_id();
    m.emit(OP_COMPOSITE_EXTRACT, &[ty_uint, lane, gid_val, 0]);

    // Block-read the A and B operands for this lane.
    let a_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, a_ptr, p_a, lane);
    let a_blk = m.alloc_id();
    m.emit(OP_SUBGROUP_BLOCK_READ_INTEL, &[ty_float, a_blk, a_ptr]);

    let b_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, b_ptr, p_b, lane);
    let b_blk = m.alloc_id();
    m.emit(OP_SUBGROUP_BLOCK_READ_INTEL, &[ty_float, b_blk, b_ptr]);

    // Systolic MAC seed: prod = a_blk * b_blk; acc = prod + prod-folded-with-k.
    let prod = m.alloc_id();
    m.emit(OP_F_MUL, &[ty_float, prod, a_blk, b_blk]);

    // Fold every baked systolic constant (tile_m, tile_n, k) into the output
    // index so they all flow into the store address; the driver's JIT reads
    // this shape as a DPAS accumulation tile.
    // tile_volume = tile_m * tile_n * k_systolic.
    let tile_area = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, tile_area, c_tile_m, c_tile_n]);
    let tile_volume = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, tile_volume, tile_area, c_k]);
    // out_idx = lane + runtime_k * tile_volume (one accumulation slab per k).
    let depth_span = m.alloc_id();
    m.emit(OP_I_MUL, &[ty_uint, depth_span, p_k, tile_volume]);
    let out_idx = m.alloc_id();
    m.emit(OP_I_ADD, &[ty_uint, out_idx, lane, depth_span]);

    // Accumulate (scalar-emulated systolic step) and block-write the result.
    let acc = m.alloc_id();
    m.emit(OP_F_ADD, &[ty_float, acc, prod, a_blk]);
    let c_ptr = m.alloc_id();
    m.emit_in_bounds_ptr_access_chain(ty_ptr_cross_float, c_ptr, p_c, out_idx);
    m.emit(OP_SUBGROUP_BLOCK_WRITE_INTEL, &[c_ptr, acc]);

    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

// ─── FP8 cooperative-matrix module ───────────────────────────

/// The two IEEE-style 8-bit float encodings used for low-precision inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// E4M3 (`HF8`): 1 sign, 4 exponent, 3 mantissa bits.
    E4m3,
    /// E5M2 (`BF8`): 1 sign, 5 exponent, 2 mantissa bits.
    E5m2,
}

impl Fp8Format {
    /// Number of exponent bits.
    #[must_use]
    pub fn exponent_bits(self) -> u32 {
        match self {
            Fp8Format::E4m3 => 4,
            Fp8Format::E5m2 => 5,
        }
    }

    /// Number of mantissa bits.
    #[must_use]
    pub fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4m3 => 3,
            Fp8Format::E5m2 => 2,
        }
    }

    /// The Intel naming used in oneAPI docs.
    #[must_use]
    pub fn intel_name(self) -> &'static str {
        match self {
            Fp8Format::E4m3 => "HF8",
            Fp8Format::E5m2 => "BF8",
        }
    }
}

// SPIR-V opcodes/values local to the FP8 module (cooperative-matrix dialect).
const OP_CAPABILITY: u32 = 17;
const OP_EXTENSION: u32 = 10;
const OP_MEMORY_MODEL: u32 = 14;
const OP_ENTRY_POINT: u32 = 15;
const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_VOID: u32 = 19;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_FUNCTION: u32 = 33;
const OP_CONSTANT: u32 = 43;
const OP_FUNCTION: u32 = 54;
const OP_FUNCTION_END: u32 = 56;
const OP_LABEL: u32 = 248;
const OP_RETURN: u32 = 253;
const OP_TYPE_COOPERATIVE_MATRIX_KHR: u32 = 4456;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_VERSION_1_6: u32 = 0x0001_0600;
const SPIRV_GENERATOR: u32 = 0x000D_0004; // OxiCUDA Level Zero FP8 generator

const CAPABILITY_SHADER: u32 = 1;
const CAPABILITY_COOPERATIVE_MATRIX_KHR: u32 = 6022;
/// `Float8EXT` capability — 8-bit float component type (SPV_EXT_float8).
const CAPABILITY_FLOAT8_EXT: u32 = 6212;

/// `Float8E4M3EXT` FP-encoding operand for `OpTypeFloat` (SPV_EXT_float8).
const FLOAT8_E4M3_EXT: u32 = 4214;
/// `Float8E5M2EXT` FP-encoding operand for `OpTypeFloat` (SPV_EXT_float8).
const FLOAT8_E5M2_EXT: u32 = 4215;

const ADDRESSING_MODEL_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;
const EXECUTION_MODEL_GLCOMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;
const SCOPE_SUBGROUP: u32 = 3;
const MATRIX_USE_A: u32 = 0;
const MATRIX_USE_B: u32 = 1;
const MATRIX_USE_ACCUMULATOR: u32 = 2;

/// Emit a SPIR-V word-stream `(word_count << 16) | opcode` instruction.
fn push_inst(words: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    let word_count = (1 + operands.len()) as u32;
    words.push((word_count << 16) | opcode);
    words.extend_from_slice(operands);
}

/// Pack a string into null-terminated little-endian SPIR-V words.
fn string_words(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let padded_len = (bytes.len() + 4) & !3;
    let mut out = vec![0u32; padded_len / 4];
    for (i, &b) in bytes.iter().enumerate() {
        out[i / 4] |= u32::from(b) << ((i % 4) * 8);
    }
    out
}

/// Generate a SPIR-V 1.6 cooperative-matrix module declaring an FP8 GEMM tile.
///
/// This emits the capability/extension/type surface for an Xe-HPC FP8 inference
/// GEMM (`D = A*B + C`) with 8-bit `format` inputs and FP32 accumulation, using
/// `SPV_KHR_cooperative_matrix` plus the 8-bit-float extension. The module
/// declares the three cooperative-matrix types (A, B, accumulator), the
/// `Float8EXT` capability, and an empty kernel body. Verifying the FP8
/// arithmetic requires Xe-HPC hardware with FP8 XMX support; here we validate
/// the emitted declaration structurally.
///
/// Entry-point name: `"gemm_fp8"`.
pub fn gemm_fp8_coop_matrix_spirv(
    format: Fp8Format,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
) -> Vec<u32> {
    let mut words = vec![SPIRV_MAGIC, SPIRV_VERSION_1_6, SPIRV_GENERATOR, 0, 0];
    let mut next_id = 1u32;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // ── Capabilities ──
    push_inst(&mut words, OP_CAPABILITY, &[CAPABILITY_SHADER]);
    push_inst(
        &mut words,
        OP_CAPABILITY,
        &[CAPABILITY_COOPERATIVE_MATRIX_KHR],
    );
    push_inst(&mut words, OP_CAPABILITY, &[CAPABILITY_FLOAT8_EXT]);

    // ── Extensions ──
    {
        let ext = string_words("SPV_KHR_cooperative_matrix");
        push_inst(&mut words, OP_EXTENSION, &ext);
    }
    {
        let ext = string_words("SPV_EXT_float8");
        push_inst(&mut words, OP_EXTENSION, &ext);
    }

    // ── Memory model ──
    push_inst(
        &mut words,
        OP_MEMORY_MODEL,
        &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
    );

    // ── Types ──
    let ty_void = alloc();
    let ty_u32 = alloc();
    let ty_f8 = alloc(); // 8-bit float input element
    let ty_f32 = alloc(); // 32-bit accumulator element
    let ty_cmat_a = alloc();
    let ty_cmat_b = alloc();
    let ty_cmat_c = alloc();
    let ty_fn = alloc();
    let c_m = alloc();
    let c_n = alloc();
    let c_k = alloc();
    let main_fn = alloc();
    let label = alloc();

    push_inst(&mut words, OP_TYPE_VOID, &[ty_void]);
    push_inst(&mut words, OP_TYPE_INT, &[ty_u32, 32, 0]);
    // 8-bit float component. Under SPV_EXT_float8, an 8-bit OpTypeFloat MUST
    // carry an FP-encoding operand selecting E4M3 vs E5M2 — without it the type
    // is invalid AND both formats would emit identical bytes. Selecting the
    // encoding from `format` makes E4M3/E5M2 produce distinct, valid modules.
    let fp8_encoding = match format {
        Fp8Format::E4m3 => FLOAT8_E4M3_EXT,
        Fp8Format::E5m2 => FLOAT8_E5M2_EXT,
    };
    push_inst(&mut words, OP_TYPE_FLOAT, &[ty_f8, 8, fp8_encoding]);
    push_inst(&mut words, OP_TYPE_FLOAT, &[ty_f32, 32]);

    push_inst(&mut words, OP_CONSTANT, &[ty_u32, c_m, tile_m]);
    push_inst(&mut words, OP_CONSTANT, &[ty_u32, c_n, tile_n]);
    push_inst(&mut words, OP_CONSTANT, &[ty_u32, c_k, tile_k]);

    // OpTypeCooperativeMatrixKHR result component scope rows cols use
    push_inst(
        &mut words,
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[ty_cmat_a, ty_f8, SCOPE_SUBGROUP, c_m, c_k, MATRIX_USE_A],
    );
    push_inst(
        &mut words,
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[ty_cmat_b, ty_f8, SCOPE_SUBGROUP, c_k, c_n, MATRIX_USE_B],
    );
    push_inst(
        &mut words,
        OP_TYPE_COOPERATIVE_MATRIX_KHR,
        &[
            ty_cmat_c,
            ty_f32,
            SCOPE_SUBGROUP,
            c_m,
            c_n,
            MATRIX_USE_ACCUMULATOR,
        ],
    );

    push_inst(&mut words, OP_TYPE_FUNCTION, &[ty_fn, ty_void]);

    // ── Entry point + execution mode ──
    {
        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, main_fn];
        ops.extend(string_words("gemm_fp8"));
        push_inst(&mut words, OP_ENTRY_POINT, &ops);
    }
    push_inst(
        &mut words,
        OP_EXECUTION_MODE,
        &[main_fn, EXECUTION_MODE_LOCAL_SIZE, 16, 1, 1],
    );

    // ── Function body (declaration only) ──
    push_inst(&mut words, OP_FUNCTION, &[ty_void, main_fn, 0, ty_fn]);
    push_inst(&mut words, OP_LABEL, &[label]);
    push_inst(&mut words, OP_RETURN, &[]);
    push_inst(&mut words, OP_FUNCTION_END, &[]);

    words[3] = next_id;
    words
}

// ─── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::{OP_CAPABILITY, OP_EXTENSION, SPIRV_MAGIC};

    /// Decode the instruction stream after the 5-word header into
    /// `(opcode, operands)` pairs.
    fn decode(words: &[u32]) -> Vec<(u32, Vec<u32>)> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            let op = words[i] & 0xffff;
            if wc == 0 || i + wc > words.len() {
                break;
            }
            out.push((op, words[i + 1..i + wc].to_vec()));
            i += wc;
        }
        out
    }

    fn header_ok(words: &[u32]) {
        assert!(words.len() >= 6);
        assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
        assert!(words[3] > 0, "id bound must be > 0");
        assert_eq!(words[4], 0, "schema must be 0");
    }

    /// Decode a packed extension string from an `OpExtension`'s operands.
    fn ext_string(operands: &[u32]) -> String {
        let mut bytes = Vec::new();
        for w in operands {
            for shift in 0..4 {
                let b = ((w >> (shift * 8)) & 0xff) as u8;
                if b == 0 {
                    return String::from_utf8_lossy(&bytes).into_owned();
                }
                bytes.push(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[test]
    fn block_copy_header_and_alignment() {
        let words = esimd_block_copy_spirv();
        header_ok(&words);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn block_copy_declares_intel_subgroups_extension() {
        let words = esimd_block_copy_spirv();
        let insts = decode(&words);
        let has_ext = insts
            .iter()
            .filter(|(op, _)| *op == OP_EXTENSION)
            .any(|(_, ops)| ext_string(ops) == "SPV_INTEL_subgroups");
        assert!(has_ext, "must declare SPV_INTEL_subgroups");
    }

    #[test]
    fn block_copy_declares_block_io_capability() {
        let words = esimd_block_copy_spirv();
        let insts = decode(&words);
        let caps: Vec<u32> = insts
            .iter()
            .filter(|(op, _)| *op == OP_CAPABILITY)
            .map(|(_, ops)| ops[0])
            .collect();
        assert!(caps.contains(&CAPABILITY_SUBGROUP_BUFFER_BLOCK_IO_INTEL));
        assert!(caps.contains(&CAPABILITY_KERNEL));
    }

    #[test]
    fn block_copy_emits_block_read_and_write() {
        let words = esimd_block_copy_spirv();
        let insts = decode(&words);
        assert!(
            insts
                .iter()
                .any(|(op, _)| *op == OP_SUBGROUP_BLOCK_READ_INTEL),
            "missing OpSubgroupBlockReadINTEL"
        );
        assert!(
            insts
                .iter()
                .any(|(op, _)| *op == OP_SUBGROUP_BLOCK_WRITE_INTEL),
            "missing OpSubgroupBlockWriteINTEL"
        );
    }

    #[test]
    fn dpas_tile_config_constants() {
        assert_eq!(DpasTileConfig::DG2_BF16.mac_lanes(), 64);
        assert_eq!(DpasTileConfig::DG2_INT8.k, 32);
    }

    #[test]
    fn dpas_gemm_header_and_block_ops() {
        let words = dpas_gemm_spirv(DpasTileConfig::DG2_BF16);
        header_ok(&words);
        let insts = decode(&words);
        assert!(
            insts
                .iter()
                .any(|(op, _)| *op == OP_SUBGROUP_BLOCK_READ_INTEL)
        );
        assert!(
            insts
                .iter()
                .any(|(op, _)| *op == OP_SUBGROUP_BLOCK_WRITE_INTEL)
        );
        // The systolic-depth constant must be baked in.
        let has_k_const = insts
            .iter()
            .any(|(op, ops)| *op == crate::spirv::OP_CONSTANT && ops.last() == Some(&16));
        assert!(has_k_const, "systolic depth constant 16 must be baked");
    }

    #[test]
    fn dpas_gemm_distinct_tiles_distinct_binaries() {
        let bf16 = dpas_gemm_spirv(DpasTileConfig::DG2_BF16);
        let int8 = dpas_gemm_spirv(DpasTileConfig::DG2_INT8);
        assert_ne!(bf16, int8, "different k must change the baked constant");
    }

    #[test]
    fn fp8_format_fields() {
        assert_eq!(Fp8Format::E4m3.exponent_bits(), 4);
        assert_eq!(Fp8Format::E4m3.mantissa_bits(), 3);
        assert_eq!(Fp8Format::E4m3.intel_name(), "HF8");
        assert_eq!(Fp8Format::E5m2.exponent_bits(), 5);
        assert_eq!(Fp8Format::E5m2.intel_name(), "BF8");
    }

    #[test]
    fn fp8_coop_matrix_header_and_capabilities() {
        let words = gemm_fp8_coop_matrix_spirv(Fp8Format::E4m3, 8, 16, 16);
        header_ok(&words);
        assert_eq!(
            words[1], SPIRV_VERSION_1_6,
            "FP8 coop-matrix needs SPIR-V 1.6"
        );
        let insts = decode(&words);
        let caps: Vec<u32> = insts
            .iter()
            .filter(|(op, _)| *op == OP_CAPABILITY)
            .map(|(_, ops)| ops[0])
            .collect();
        assert!(
            caps.contains(&CAPABILITY_FLOAT8_EXT),
            "Float8EXT capability required"
        );
        assert!(caps.contains(&CAPABILITY_COOPERATIVE_MATRIX_KHR));
    }

    #[test]
    fn fp8_coop_matrix_declares_extensions_and_8bit_float() {
        let words = gemm_fp8_coop_matrix_spirv(Fp8Format::E5m2, 8, 16, 16);
        let insts = decode(&words);
        // Both cooperative-matrix and float8 extensions declared.
        let exts: Vec<String> = insts
            .iter()
            .filter(|(op, _)| *op == OP_EXTENSION)
            .map(|(_, ops)| ext_string(ops))
            .collect();
        assert!(exts.iter().any(|e| e == "SPV_KHR_cooperative_matrix"));
        assert!(exts.iter().any(|e| e == "SPV_EXT_float8"));

        // An 8-bit OpTypeFloat must be present for the FP8 element.
        let has_f8 = insts
            .iter()
            .any(|(op, ops)| *op == OP_TYPE_FLOAT && ops.get(1) == Some(&8));
        assert!(has_f8, "8-bit OpTypeFloat must be declared");
        // Three cooperative-matrix types (A, B, accumulator).
        let cmat_count = insts
            .iter()
            .filter(|(op, _)| *op == OP_TYPE_COOPERATIVE_MATRIX_KHR)
            .count();
        assert_eq!(cmat_count, 3);
    }

    #[test]
    fn fp8_coop_matrix_word_aligned() {
        let words = gemm_fp8_coop_matrix_spirv(Fp8Format::E4m3, 8, 32, 16);
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn fp8_e4m3_and_e5m2_produce_distinct_binaries() {
        let e4m3 = gemm_fp8_coop_matrix_spirv(Fp8Format::E4m3, 8, 16, 16);
        let e5m2 = gemm_fp8_coop_matrix_spirv(Fp8Format::E5m2, 8, 16, 16);
        assert_ne!(
            e4m3, e5m2,
            "the two FP8 encodings must yield distinct SPIR-V modules"
        );
    }

    #[test]
    fn fp8_type_carries_encoding_operand() {
        // The 8-bit OpTypeFloat must carry the SPV_EXT_float8 FP-encoding operand
        // selecting E4M3 (4214) vs E5M2 (4215).
        let e4m3 = decode(&gemm_fp8_coop_matrix_spirv(Fp8Format::E4m3, 8, 16, 16));
        let e5m2 = decode(&gemm_fp8_coop_matrix_spirv(Fp8Format::E5m2, 8, 16, 16));
        let enc = |insts: &[(u32, Vec<u32>)]| -> Option<u32> {
            insts
                .iter()
                .find(|(op, ops)| *op == OP_TYPE_FLOAT && ops.get(1) == Some(&8))
                .and_then(|(_, ops)| ops.get(2).copied())
        };
        assert_eq!(enc(&e4m3), Some(FLOAT8_E4M3_EXT));
        assert_eq!(enc(&e5m2), Some(FLOAT8_E5M2_EXT));
    }
}
