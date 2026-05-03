//! Trivial placeholder compute shader (`void main() {}`).

use super::builder::SpvModule;
use super::consts::{
    CAPABILITY_SHADER, EXECUTION_MODEL_GLCOMPUTE, FUNCTION_CONTROL_NONE, OP_ENTRY_POINT,
};

/// Build a minimal valid compute shader: `void main() {}` with `LocalSize(1,1,1)`.
pub fn trivial_compute_shader() -> Vec<u32> {
    let mut m = SpvModule::new();

    let id_main_fn = m.alloc_id();
    let id_void = m.alloc_id();
    let id_void_fn = m.alloc_id();
    let id_label = m.alloc_id();

    m.emit_capability(CAPABILITY_SHADER);
    m.emit_memory_model();

    let mut entry_words = vec![EXECUTION_MODEL_GLCOMPUTE, id_main_fn];
    entry_words.extend(SpvModule::string_words("main"));
    m.emit(OP_ENTRY_POINT, &entry_words);

    m.emit_execution_mode_local_size(id_main_fn, 1, 1, 1);

    m.emit_type_void(id_void);
    m.emit_type_function(id_void_fn, id_void, &[]);

    m.emit_function(id_void, id_main_fn, FUNCTION_CONTROL_NONE, id_void_fn);
    m.emit_label(id_label);
    m.emit_return();
    m.emit_function_end();

    m.finalize()
}

/// Return the trivial compute shader as a byte slice.
pub fn trivial_compute_shader_bytes() -> Vec<u8> {
    trivial_compute_shader()
        .iter()
        .flat_map(|w| w.to_ne_bytes())
        .collect()
}
