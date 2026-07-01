//! Naga WGSL parse + validation tests for every CPU-testable shader generator.
//!
//! For each generated WGSL string we run two naga passes:
//!   1. `naga::front::wgsl::parse_str` — checks tokenisation + AST validity.
//!   2. `naga::valid::Validator::new(ValidationFlags::all(), caps).validate` —
//!      checks types, bindings, control-flow, etc.
//!
//! Skipped generators (with explanation):
//!   * `subgroup_reduction_wgsl` (both `enable subgroups;` and
//!     `enable chromium_experimental_subgroups;` variants) — naga 29.0.3
//!     front-end rejects `enable subgroups;` at parse time; this is a known
//!     naga gap for the WGSL subgroups extension (Chromium experimental path).
//!     The generators are tested structurally in `shader_ext` module tests.

use naga::valid::{Capabilities, ValidationFlags};

use crate::fft::{fft_bitreverse_wgsl, fft_stage_wgsl};
use crate::shader::{
    attention_wgsl, batched_gemm_wgsl, binary_wgsl, conv2d_wgsl, elementwise_wgsl, gemm_wgsl,
    gemm_wgsl_f16, reduction_final_wgsl, reduction_nd_wgsl, reduction_wgsl,
};
use crate::shader_ext::{
    ScanKind, f64_emul_add_wgsl, layernorm_wgsl, scan_wgsl, softmax_wgsl, transpose_wgsl,
};

/// Parse and fully validate a WGSL source string with the given capabilities.
///
/// Panics with a descriptive message on the first failure, including which
/// generator produced the bad source and the naga error.
fn assert_wgsl_valid(label: &str, src: &str, caps: Capabilities) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed for `{label}`: {e}"));
    let mut validator = naga::valid::Validator::new(ValidationFlags::all(), caps);
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("WGSL validation failed for `{label}`: {e}"));
}

// ── shader.rs generators ─────────────────────────────────────────────────────

#[test]
fn naga_validates_gemm_wgsl() {
    assert_wgsl_valid("gemm_wgsl(8)", &gemm_wgsl(8), Capabilities::empty());
}

#[test]
fn naga_validates_gemm_wgsl_tile16() {
    assert_wgsl_valid("gemm_wgsl(16)", &gemm_wgsl(16), Capabilities::empty());
}

#[test]
fn naga_validates_gemm_wgsl_f16() {
    // The f16 shader requires the SHADER_FLOAT16 capability.
    assert_wgsl_valid(
        "gemm_wgsl_f16(8)",
        &gemm_wgsl_f16(8),
        Capabilities::SHADER_FLOAT16,
    );
}

#[test]
fn naga_validates_batched_gemm_wgsl() {
    assert_wgsl_valid(
        "batched_gemm_wgsl(8)",
        &batched_gemm_wgsl(8),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_elementwise_wgsl_relu() {
    assert_wgsl_valid(
        "elementwise_wgsl(relu)",
        &elementwise_wgsl("relu"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_elementwise_wgsl_sigmoid() {
    assert_wgsl_valid(
        "elementwise_wgsl(sigmoid)",
        &elementwise_wgsl("sigmoid"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_elementwise_wgsl_tanh() {
    assert_wgsl_valid(
        "elementwise_wgsl(tanh)",
        &elementwise_wgsl("tanh"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_elementwise_wgsl_neg() {
    assert_wgsl_valid(
        "elementwise_wgsl(neg)",
        &elementwise_wgsl("neg"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_binary_wgsl_add() {
    assert_wgsl_valid(
        "binary_wgsl(add)",
        &binary_wgsl("add"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_binary_wgsl_pow() {
    assert_wgsl_valid(
        "binary_wgsl(pow)",
        &binary_wgsl("pow"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_wgsl_sum() {
    assert_wgsl_valid(
        "reduction_wgsl(sum)",
        &reduction_wgsl("sum"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_wgsl_max() {
    assert_wgsl_valid(
        "reduction_wgsl(max)",
        &reduction_wgsl("max"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_wgsl_min() {
    assert_wgsl_valid(
        "reduction_wgsl(min)",
        &reduction_wgsl("min"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_nd_wgsl_sum() {
    assert_wgsl_valid(
        "reduction_nd_wgsl(sum)",
        &reduction_nd_wgsl("sum"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_nd_wgsl_mean() {
    assert_wgsl_valid(
        "reduction_nd_wgsl(mean)",
        &reduction_nd_wgsl("mean"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_reduction_final_wgsl() {
    assert_wgsl_valid(
        "reduction_final_wgsl(sum)",
        &reduction_final_wgsl("sum"),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_conv2d_wgsl() {
    // Small but valid dims: 1-batch, 1-channel, 4×4 input, 1 filter, 3×3
    // kernel, 2×2 output (valid conv, no padding, stride 1).
    // The binding previously named `filter` (a WGSL reserved keyword) was
    // renamed to `kernel_w`; this test confirms the fixed shader parses.
    let src = conv2d_wgsl(1, 1, 4, 4, 1, 3, 3, 2, 2, 1, 1, 0, 0);
    assert_wgsl_valid(
        "conv2d_wgsl(1,1,4,4,1,3,3,2,2,1,1,0,0)",
        &src,
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_conv2d_wgsl_with_padding() {
    // Padded conv: 1-batch, 3-channel, 8×8 input, 4 filters, 3×3 kernel,
    // 8×8 output (same-pad with pad=1, stride=1).
    let src = conv2d_wgsl(1, 3, 8, 8, 4, 3, 3, 8, 8, 1, 1, 1, 1);
    assert_wgsl_valid(
        "conv2d_wgsl(1,3,8,8,4,3,3,8,8,1,1,1,1)",
        &src,
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_attention_wgsl_non_causal() {
    let src = attention_wgsl(2, 4, 4, 8, 0.5_f32, false);
    assert_wgsl_valid(
        "attention_wgsl(2,4,4,8,0.5,causal=false)",
        &src,
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_attention_wgsl_causal() {
    let src = attention_wgsl(2, 4, 4, 8, 0.5_f32, true);
    assert_wgsl_valid(
        "attention_wgsl(2,4,4,8,0.5,causal=true)",
        &src,
        Capabilities::empty(),
    );
}

// ── shader_ext.rs generators ─────────────────────────────────────────────────

#[test]
fn naga_validates_transpose_wgsl() {
    assert_wgsl_valid(
        "transpose_wgsl(8)",
        &transpose_wgsl(8),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_transpose_wgsl_tile16() {
    assert_wgsl_valid(
        "transpose_wgsl(16)",
        &transpose_wgsl(16),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_softmax_wgsl() {
    assert_wgsl_valid("softmax_wgsl()", &softmax_wgsl(), Capabilities::empty());
}

#[test]
fn naga_validates_scan_wgsl_inclusive() {
    assert_wgsl_valid(
        "scan_wgsl(256, Inclusive)",
        &scan_wgsl(256, ScanKind::Inclusive),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_scan_wgsl_exclusive() {
    assert_wgsl_valid(
        "scan_wgsl(256, Exclusive)",
        &scan_wgsl(256, ScanKind::Exclusive),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_scan_wgsl_block64() {
    assert_wgsl_valid(
        "scan_wgsl(64, Inclusive)",
        &scan_wgsl(64, ScanKind::Inclusive),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_layernorm_wgsl() {
    assert_wgsl_valid(
        "layernorm_wgsl(1e-5)",
        &layernorm_wgsl(1e-5),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_layernorm_wgsl_small_eps() {
    assert_wgsl_valid(
        "layernorm_wgsl(1e-8)",
        &layernorm_wgsl(1e-8),
        Capabilities::empty(),
    );
}

// SKIPPED: subgroup_reduction_wgsl (both standard and chromium_experimental
// variants) — naga 29.0.3 front-end rejects `enable subgroups;` at parse
// time. This is a known naga gap for the WGSL subgroups extension; the
// generators are covered by structural substring-assert tests in shader_ext.

#[test]
fn naga_validates_f64_emul_add_wgsl() {
    assert_wgsl_valid(
        "f64_emul_add_wgsl()",
        &f64_emul_add_wgsl(),
        Capabilities::empty(),
    );
}

// ── fft.rs generators ────────────────────────────────────────────────────────

#[test]
fn naga_validates_fft_bitreverse_wgsl() {
    assert_wgsl_valid(
        "fft_bitreverse_wgsl()",
        &fft_bitreverse_wgsl(),
        Capabilities::empty(),
    );
}

#[test]
fn naga_validates_fft_stage_wgsl() {
    assert_wgsl_valid("fft_stage_wgsl()", &fft_stage_wgsl(), Capabilities::empty());
}
