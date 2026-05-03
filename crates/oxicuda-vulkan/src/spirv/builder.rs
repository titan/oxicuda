//! Lightweight SPIR-V word-stream builder ([`SpvModule`]).
//!
//! Emits valid SPIR-V instructions for simple compute shaders without
//! pulling in a full compiler.  The convenience emitter methods are
//! `pub(super)` so they can be reused by every shader generator inside
//! [`crate::spirv`].

use super::consts::{
    ADDRESSING_MODEL_LOGICAL, EXECUTION_MODE_LOCAL_SIZE, EXECUTION_MODEL_GLCOMPUTE,
    LOOP_CONTROL_NONE, MEMORY_MODEL_GLSL450, OP_ACCESS_CHAIN, OP_BRANCH, OP_BRANCH_CONDITIONAL,
    OP_CAPABILITY, OP_CONSTANT, OP_CONTROL_BARRIER, OP_DECORATE, OP_ENTRY_POINT, OP_EXECUTION_MODE,
    OP_EXT_INST, OP_EXT_INST_IMPORT, OP_FUNCTION, OP_FUNCTION_END, OP_LABEL, OP_LOAD,
    OP_LOOP_MERGE, OP_MEMBER_DECORATE, OP_MEMORY_MODEL, OP_RETURN, OP_SELECTION_MERGE, OP_STORE,
    OP_TYPE_ARRAY, OP_TYPE_BOOL, OP_TYPE_FLOAT, OP_TYPE_FUNCTION, OP_TYPE_INT, OP_TYPE_POINTER,
    OP_TYPE_RUNTIME_ARRAY, OP_TYPE_STRUCT, OP_TYPE_VECTOR, OP_TYPE_VOID, OP_VARIABLE,
    SELECTION_CONTROL_NONE, SPIRV_GENERATOR, SPIRV_MAGIC, SPIRV_VERSION_1_2,
};

/// Lightweight SPIR-V word-stream builder.
///
/// Emits valid SPIR-V instructions for simple compute shaders without
/// pulling in a full compiler.
pub struct SpvModule {
    words: Vec<u32>,
    /// Next available result ID.
    id_bound: u32,
}

impl SpvModule {
    /// Create a new module targeting SPIR-V `version`.
    pub fn with_version(version: u32) -> Self {
        let words = vec![SPIRV_MAGIC, version, SPIRV_GENERATOR, 0, 0];
        Self { words, id_bound: 1 }
    }

    /// Create a new module with a placeholder header (SPIR-V 1.2).
    pub fn new() -> Self {
        Self::with_version(SPIRV_VERSION_1_2)
    }

    /// Allocate a fresh result ID.
    pub fn alloc_id(&mut self) -> u32 {
        let id = self.id_bound;
        self.id_bound += 1;
        id
    }

    /// Emit a SPIR-V instruction.
    pub fn emit(&mut self, opcode: u32, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u32;
        self.words.push((word_count << 16) | opcode);
        self.words.extend_from_slice(operands);
    }

    /// Emit a string as null-terminated UTF-8 packed into 32-bit words.
    pub fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        let padded_len = (bytes.len() + 4) & !3;
        let mut out = vec![0u32; padded_len / 4];
        for (i, &b) in bytes.iter().enumerate() {
            out[i / 4] |= (b as u32) << ((i % 4) * 8);
        }
        out
    }

    /// Finalise the module: patch the ID bound and return the word vector.
    pub fn finalize(mut self) -> Vec<u32> {
        self.words[3] = self.id_bound;
        self.words
    }

    // ── Convenience emitters ─────────────────────────────────

    pub(super) fn emit_capability(&mut self, cap: u32) {
        self.emit(OP_CAPABILITY, &[cap]);
    }

    pub(super) fn emit_ext_inst_import(&mut self, id: u32, name: &str) {
        let mut ops = vec![id];
        ops.extend(Self::string_words(name));
        self.emit(OP_EXT_INST_IMPORT, &ops);
    }

    pub(super) fn emit_memory_model(&mut self) {
        self.emit(
            OP_MEMORY_MODEL,
            &[ADDRESSING_MODEL_LOGICAL, MEMORY_MODEL_GLSL450],
        );
    }

    pub(super) fn emit_entry_point(&mut self, func_id: u32, name: &str, interfaces: &[u32]) {
        let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, func_id];
        ops.extend(Self::string_words(name));
        ops.extend_from_slice(interfaces);
        self.emit(OP_ENTRY_POINT, &ops);
    }

    pub(super) fn emit_execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.emit(
            OP_EXECUTION_MODE,
            &[func_id, EXECUTION_MODE_LOCAL_SIZE, x, y, z],
        );
    }

    pub(super) fn emit_decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(operands);
        self.emit(OP_DECORATE, &ops);
    }

    pub(super) fn emit_member_decorate(
        &mut self,
        ty: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let mut ops = vec![ty, member, decoration];
        ops.extend_from_slice(operands);
        self.emit(OP_MEMBER_DECORATE, &ops);
    }

    pub(super) fn emit_type_void(&mut self, id: u32) {
        self.emit(OP_TYPE_VOID, &[id]);
    }

    pub(super) fn emit_type_bool(&mut self, id: u32) {
        self.emit(OP_TYPE_BOOL, &[id]);
    }

    pub(super) fn emit_type_int(&mut self, id: u32, width: u32, signedness: u32) {
        self.emit(OP_TYPE_INT, &[id, width, signedness]);
    }

    pub(super) fn emit_type_float(&mut self, id: u32, width: u32) {
        self.emit(OP_TYPE_FLOAT, &[id, width]);
    }

    pub(super) fn emit_type_vector(&mut self, id: u32, component: u32, count: u32) {
        self.emit(OP_TYPE_VECTOR, &[id, component, count]);
    }

    pub(super) fn emit_type_runtime_array(&mut self, id: u32, element: u32) {
        self.emit(OP_TYPE_RUNTIME_ARRAY, &[id, element]);
    }

    pub(super) fn emit_type_struct(&mut self, id: u32, members: &[u32]) {
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        self.emit(OP_TYPE_STRUCT, &ops);
    }

    pub(super) fn emit_type_pointer(&mut self, id: u32, storage_class: u32, pointee: u32) {
        self.emit(OP_TYPE_POINTER, &[id, storage_class, pointee]);
    }

    pub(super) fn emit_type_function(&mut self, id: u32, return_type: u32, params: &[u32]) {
        let mut ops = vec![id, return_type];
        ops.extend_from_slice(params);
        self.emit(OP_TYPE_FUNCTION, &ops);
    }

    pub(super) fn emit_constant_u32(&mut self, ty: u32, id: u32, value: u32) {
        self.emit(OP_CONSTANT, &[ty, id, value]);
    }

    pub(super) fn emit_constant_f32(&mut self, ty: u32, id: u32, value: f32) {
        self.emit(OP_CONSTANT, &[ty, id, value.to_bits()]);
    }

    pub(super) fn emit_variable(&mut self, ty: u32, id: u32, storage_class: u32) {
        self.emit(OP_VARIABLE, &[ty, id, storage_class]);
    }

    pub(super) fn emit_load(&mut self, result_ty: u32, result: u32, pointer: u32) {
        self.emit(OP_LOAD, &[result_ty, result, pointer]);
    }

    pub(super) fn emit_store(&mut self, pointer: u32, value: u32) {
        self.emit(OP_STORE, &[pointer, value]);
    }

    pub(super) fn emit_access_chain(
        &mut self,
        result_ty: u32,
        result: u32,
        base: u32,
        indices: &[u32],
    ) {
        let mut ops = vec![result_ty, result, base];
        ops.extend_from_slice(indices);
        self.emit(OP_ACCESS_CHAIN, &ops);
    }

    pub(super) fn emit_function(&mut self, result_ty: u32, result: u32, control: u32, fn_ty: u32) {
        self.emit(OP_FUNCTION, &[result_ty, result, control, fn_ty]);
    }

    pub(super) fn emit_label(&mut self, id: u32) {
        self.emit(OP_LABEL, &[id]);
    }

    pub(super) fn emit_return(&mut self) {
        self.emit(OP_RETURN, &[]);
    }

    pub(super) fn emit_function_end(&mut self) {
        self.emit(OP_FUNCTION_END, &[]);
    }

    pub(super) fn emit_branch(&mut self, target: u32) {
        self.emit(OP_BRANCH, &[target]);
    }

    pub(super) fn emit_branch_conditional(&mut self, cond: u32, true_label: u32, false_label: u32) {
        self.emit(OP_BRANCH_CONDITIONAL, &[cond, true_label, false_label]);
    }

    pub(super) fn emit_selection_merge(&mut self, merge_label: u32) {
        self.emit(OP_SELECTION_MERGE, &[merge_label, SELECTION_CONTROL_NONE]);
    }

    pub(super) fn emit_loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.emit(
            OP_LOOP_MERGE,
            &[merge_label, continue_label, LOOP_CONTROL_NONE],
        );
    }

    pub(super) fn emit_glsl_ext(
        &mut self,
        glsl_id: u32,
        result_ty: u32,
        result: u32,
        ext: u32,
        args: &[u32],
    ) {
        let mut ops = vec![result_ty, result, glsl_id, ext];
        ops.extend_from_slice(args);
        self.emit(OP_EXT_INST, &ops);
    }

    pub(super) fn emit_type_array(&mut self, id: u32, element: u32, length: u32) {
        self.emit(OP_TYPE_ARRAY, &[id, element, length]);
    }

    pub(super) fn emit_control_barrier(&mut self, execution: u32, memory: u32, semantics: u32) {
        self.emit(OP_CONTROL_BARRIER, &[execution, memory, semantics]);
    }
}

impl Default for SpvModule {
    fn default() -> Self {
        Self::new()
    }
}
