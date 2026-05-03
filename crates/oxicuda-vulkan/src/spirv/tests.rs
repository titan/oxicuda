//! Unit tests for the SPIR-V compute-shader generators.

#![cfg(test)]

use oxicuda_backend::{BinaryOp, ReduceOp, UnaryOp};

use super::{
    SPIRV_MAGIC, SPIRV_VERSION_1_3, SpvModule, attention_spirv, batched_gemm_compute_shader,
    binary_compute_shader, conv2d_spirv, gemm_compute_shader, reduce_compute_shader,
    trivial_compute_shader, trivial_compute_shader_bytes, unary_compute_shader,
};

#[test]
fn placeholder_spv_valid_magic() {
    let words = trivial_compute_shader();
    assert!(!words.is_empty());
    assert_eq!(words[0], SPIRV_MAGIC);
}

#[test]
fn placeholder_spv_word_aligned() {
    let bytes = trivial_compute_shader_bytes();
    assert_eq!(bytes.len() % 4, 0);
}

#[test]
fn placeholder_spv_version_and_schema() {
    let words = trivial_compute_shader();
    assert!(words.len() >= 5);
    assert!(words[1] >= 0x0001_0000);
    assert_eq!(words[4], 0);
}

#[test]
fn placeholder_spv_nonzero_bound() {
    let words = trivial_compute_shader();
    assert!(words[3] > 0);
}

#[test]
fn spv_module_id_allocation_is_monotonic() {
    let mut m = SpvModule::new();
    let id1 = m.alloc_id();
    let id2 = m.alloc_id();
    assert!(id2 > id1);
}

#[test]
fn string_words_null_terminated() {
    let words = SpvModule::string_words("abc");
    assert!(!words.is_empty());
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(bytes[0], b'a');
    assert_eq!(bytes[1], b'b');
    assert_eq!(bytes[2], b'c');
    assert_eq!(bytes[3], 0);
}

#[test]
fn string_words_empty_string() {
    let words = SpvModule::string_words("");
    assert!(!words.is_empty());
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert_eq!(bytes[0], 0);
}

// ── Compute shader generation ────────────────────────────

fn check_valid_spirv(words: &[u32]) {
    assert!(words.len() >= 5, "too short for SPIR-V header");
    assert_eq!(words[0], SPIRV_MAGIC, "bad magic");
    assert!(words[3] > 0, "ID bound must be > 0");
    assert_eq!(words[4], 0, "schema must be 0");
}

#[test]
fn unary_shader_all_ops() {
    let ops = [
        UnaryOp::Relu,
        UnaryOp::Sigmoid,
        UnaryOp::Tanh,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Abs,
        UnaryOp::Neg,
    ];
    for op in ops {
        let words = unary_compute_shader(op);
        check_valid_spirv(&words);
        assert_eq!(words[1], SPIRV_VERSION_1_3, "op {op:?} wrong version");
    }
}

#[test]
fn binary_shader_all_ops() {
    let ops = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Max,
        BinaryOp::Min,
    ];
    for op in ops {
        let words = binary_compute_shader(op);
        check_valid_spirv(&words);
    }
}

#[test]
fn reduce_shader_all_ops() {
    let ops = [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min, ReduceOp::Mean];
    for op in ops {
        let words = reduce_compute_shader(op);
        check_valid_spirv(&words);
    }
}

#[test]
fn gemm_shader_valid() {
    let words = gemm_compute_shader();
    check_valid_spirv(&words);
    assert_eq!(words[1], SPIRV_VERSION_1_3);
}

#[test]
fn batched_gemm_shader_valid() {
    let words = batched_gemm_compute_shader();
    check_valid_spirv(&words);
    assert_eq!(words[1], SPIRV_VERSION_1_3);
    // Batched GEMM shader must be larger than regular GEMM (extra batch logic).
    let gemm_words = gemm_compute_shader();
    assert!(
        words.len() > gemm_words.len(),
        "batched_gemm ({}) should be larger than gemm ({})",
        words.len(),
        gemm_words.len()
    );
}

#[test]
fn batched_gemm_shader_contains_expected_structure() {
    let words = batched_gemm_compute_shader();
    // Magic number is correct.
    assert_eq!(words[0], 0x07230203);
    // ID bound is positive.
    assert!(words[3] > 0);
    // Schema is 0.
    assert_eq!(words[4], 0);
    // Must contain at least the capability, memory model, and entry point.
    assert!(words.len() > 50, "shader too small: {}", words.len());
}

#[test]
fn conv2d_shader_valid() {
    // 1×1 identity-style convolution
    let words = conv2d_spirv(1, 1, 4, 4, 1, 1, 1, 4, 4, 1, 1, 0, 0);
    check_valid_spirv(&words);
    assert_eq!(words[1], SPIRV_VERSION_1_3);
    assert!(
        words.len() > 100,
        "conv2d shader too small: {}",
        words.len()
    );
}

#[test]
fn conv2d_shader_with_padding() {
    let words = conv2d_spirv(2, 3, 8, 8, 16, 3, 3, 8, 8, 1, 1, 1, 1);
    check_valid_spirv(&words);
    assert!(words.len() > 100);
}

#[test]
fn attention_shader_valid() {
    let words = attention_spirv(2, 4, 4, 8, 0.125, false);
    check_valid_spirv(&words);
    assert_eq!(words[1], SPIRV_VERSION_1_3);
    assert!(
        words.len() > 100,
        "attention shader too small: {}",
        words.len()
    );
}

#[test]
fn attention_shader_causal() {
    let words = attention_spirv(4, 16, 16, 64, 0.125, true);
    check_valid_spirv(&words);
    assert!(words.len() > 100);
    // Causal shader should be larger (extra branching)
    let non_causal = attention_spirv(4, 16, 16, 64, 0.125, false);
    assert!(
        words.len() > non_causal.len(),
        "causal {} should be larger than non-causal {}",
        words.len(),
        non_causal.len()
    );
}
