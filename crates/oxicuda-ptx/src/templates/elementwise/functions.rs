//! Free helper functions for the elementwise template module.
//!
//! Provides PTX hex literal helpers and scalar parameter type promotion,
//! plus the test suite covering the elementwise template surface.
//!
//! Refactored with [SplitRS](https://github.com/cool-japan/splitrs).

use crate::ir::PtxType;

/// Returns the IEEE 754 hex literal for 1.0 in the given precision.
pub(super) const fn float_one_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d3FF0000000000000",
        _ => "0f3F800000",
    }
}
/// Returns the IEEE 754 hex literal for 2.0 in the given precision.
pub(super) const fn float_two_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d4000000000000000",
        _ => "0f40000000",
    }
}
/// Returns the IEEE 754 hex literal for 0.0 in the given precision.
pub(super) const fn float_zero_literal(ty: PtxType) -> &'static str {
    match ty {
        PtxType::F64 => "0d0000000000000000",
        _ => "0f00000000",
    }
}
/// Returns the scalar parameter type matching the given float precision.
///
/// For F16 and BF16, scalar parameters are passed as F32 (promoted).
pub(super) const fn scalar_param_type(ty: PtxType) -> PtxType {
    match ty {
        PtxType::F16 | PtxType::BF16 => PtxType::F32,
        other => other,
    }
}
#[cfg(test)]
mod tests {
    use super::super::elementwisetemplate_type::ElementwiseTemplate;
    use super::super::types::ElementwiseOp;
    use super::*;
    use crate::arch::SmVersion;
    #[test]
    fn elementwise_op_names() {
        assert_eq!(ElementwiseOp::Add.as_str(), "add");
        assert_eq!(ElementwiseOp::Relu.as_str(), "relu");
        assert_eq!(ElementwiseOp::FusedScaleAdd.as_str(), "fused_scale_add");
    }
    #[test]
    fn elementwise_op_classification() {
        assert!(ElementwiseOp::Add.is_binary());
        assert!(ElementwiseOp::Sub.is_binary());
        assert!(!ElementwiseOp::Relu.is_binary());
        assert!(!ElementwiseOp::Sigmoid.is_binary());
        assert!(ElementwiseOp::Scale.needs_scalar());
        assert!(ElementwiseOp::FusedScaleAdd.needs_scalar());
        assert!(!ElementwiseOp::Add.needs_scalar());
    }
    #[test]
    fn kernel_name_format() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::F32, SmVersion::Sm80);
        assert_eq!(t.kernel_name(), "elementwise_add_f32");
        let t2 = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F16, SmVersion::Sm90);
        assert_eq!(t2.kernel_name(), "elementwise_relu_f16");
    }
    #[test]
    fn invalid_precision_rejected() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::U32, SmVersion::Sm80);
        let result = t.generate();
        assert!(result.is_err());
    }
    #[test]
    fn generate_add_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Add, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate add kernel");
        assert!(ptx.contains(".entry elementwise_add_f32"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains("add.f32"));
    }
    #[test]
    fn generate_relu_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate relu kernel");
        assert!(ptx.contains(".entry elementwise_relu_f32"));
        assert!(ptx.contains("max.f32"));
    }
    #[test]
    fn generate_sigmoid_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Sigmoid, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate sigmoid kernel");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }
    #[test]
    fn generate_gelu_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Gelu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("should generate gelu kernel");
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains(".entry elementwise_gelu_f32"));
    }
    /// `ReLU` must use `max` (not `setp`/`selp`, not `sin` or other wrong ops).
    /// The implementation emits `max.f32 %f_y, %f_x, 0f00000000`.
    #[test]
    fn test_relu_ptx_correct_arithmetic() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("relu PTX generation failed");
        assert!(ptx.contains("max.f32"), "relu must emit max.f32");
        assert!(ptx.contains("0f00000000"), "relu must compare against 0.0");
        assert!(!ptx.contains("sin.approx"), "relu must not emit sin");
        assert!(!ptx.contains("cos.approx"), "relu must not emit cos");
        assert!(!ptx.contains("ex2.approx"), "relu must not use exp");
        assert!(!ptx.contains("rcp.approx"), "relu must not use rcp");
    }
    /// Sigmoid must contain: neg (for -x), ex2.approx (exp via base-2),
    /// rcp.approx (reciprocal for 1/(1+exp(-x))), and add (for +1.0).
    /// Must NOT contain wrong operations.
    #[test]
    fn test_sigmoid_ptx_contains_exp_and_rcp() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Sigmoid, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("sigmoid PTX generation failed");
        assert!(ptx.contains("neg.f32"), "sigmoid must negate input");
        assert!(
            ptx.contains("ex2.approx.f32"),
            "sigmoid must use ex2.approx for exp"
        );
        assert!(ptx.contains("0f3FB8AA3B"), "sigmoid must scale by log2(e)");
        assert!(
            ptx.contains("rcp.approx.f32"),
            "sigmoid must use rcp.approx for 1/denom"
        );
        assert!(
            ptx.contains("0f3F800000"),
            "sigmoid must add 1.0 to denominator"
        );
        assert!(!ptx.contains("sin.approx"), "sigmoid must not emit sin");
        assert!(
            !ptx.contains("max.f32"),
            "sigmoid must not use max (relu op)"
        );
    }
    /// GELU uses tanh approximation: check for the three key constants
    /// (0.044715, sqrt(2/pi), 2.0) and the ex2+rcp pattern for tanh-via-sigmoid.
    #[test]
    fn test_gelu_ptx_contains_tanh_approximation() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Gelu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("gelu PTX generation failed");
        assert!(
            ptx.contains("0f3D372713"),
            "gelu must use 0.044715 constant"
        );
        assert!(
            ptx.contains("0f3F4C422A"),
            "gelu must use sqrt(2/pi) constant"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "gelu must use ex2.approx for tanh approximation"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "gelu must use rcp.approx inside tanh"
        );
        assert!(!ptx.contains("sin.approx"), "gelu must not emit sin");
    }
    /// Tanh (implemented as `2*sigmoid(2x)-1`) must contain ex2.approx,
    /// rcp.approx, the 2.0 constant, and the subtract of 1.0.
    #[test]
    fn test_tanh_ptx_contains_exp_instructions() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Tanh, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("tanh PTX generation failed");
        assert!(
            ptx.contains("ex2.approx.f32"),
            "tanh must use ex2.approx for exp"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "tanh must use rcp.approx in sigmoid step"
        );
        assert!(ptx.contains("0f40000000"), "tanh must scale by 2.0");
        assert!(
            ptx.contains("sub.f32"),
            "tanh must subtract 1.0 for tanh formula"
        );
        assert!(!ptx.contains("sin.approx"), "tanh must not emit sin");
    }
    /// `SiLU` (`x * sigmoid(x)`) must contain both multiplication and the
    /// sigmoid sub-pattern (ex2.approx + rcp.approx).
    #[test]
    fn test_silu_ptx_contains_mul_and_sigmoid() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Silu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("silu PTX generation failed");
        assert!(
            ptx.contains("ex2.approx.f32"),
            "silu must use ex2.approx for sigmoid"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "silu must use rcp.approx for sigmoid"
        );
        assert!(
            ptx.contains("mul.f32"),
            "silu must multiply x by sigmoid(x)"
        );
        assert!(!ptx.contains("sin.approx"), "silu must not emit sin");
        assert!(!ptx.contains("max.f32"), "silu must not use relu max");
    }
    /// Every generated elementwise kernel must have valid PTX structural headers:
    /// `.version`, `.target`, and `.entry`.
    #[test]
    fn test_elementwise_ptx_has_valid_headers() {
        let ops_and_types = [
            (ElementwiseOp::Add, PtxType::F32),
            (ElementwiseOp::Relu, PtxType::F32),
            (ElementwiseOp::Sigmoid, PtxType::F32),
            (ElementwiseOp::Gelu, PtxType::F32),
            (ElementwiseOp::Tanh, PtxType::F32),
            (ElementwiseOp::Silu, PtxType::F32),
            (ElementwiseOp::Neg, PtxType::F32),
            (ElementwiseOp::Exp, PtxType::F32),
            (ElementwiseOp::Log, PtxType::F32),
        ];
        for (op, ty) in ops_and_types {
            let t = ElementwiseTemplate::new(op, ty, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX generation failed for {op:?}: {e}"));
            assert!(
                ptx.contains(".version"),
                "PTX for {op:?} must have .version header"
            );
            assert!(
                ptx.contains(".target"),
                "PTX for {op:?} must have .target header"
            );
            assert!(
                ptx.contains(".entry"),
                "PTX for {op:?} must have .entry directive"
            );
        }
    }
    /// CPU reference for `ReLU`: `max(0, x)`.
    fn cpu_relu_f32(x: f32) -> f32 {
        x.max(0.0)
    }
    /// CPU reference for sigmoid: 1 / (1 + exp(-x)).
    fn cpu_sigmoid_f32(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    /// CPU reference for GELU (tanh approximation matching PTX):
    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    fn cpu_gelu_f32(x: f32) -> f32 {
        let k0: f32 = 0.797_884_6;
        let k1: f32 = 0.044_715;
        let inner = k0 * k1.mul_add(x * x * x, x);
        0.5 * x * (1.0 + inner.tanh())
    }
    /// CPU reference for tanh: `x.tanh()`.
    fn cpu_tanh_f32(x: f32) -> f32 {
        x.tanh()
    }
    /// CPU reference for `SiLU`: `x * sigmoid(x)`.
    fn cpu_silu_f32(x: f32) -> f32 {
        x * cpu_sigmoid_f32(x)
    }
    #[test]
    fn relu_precision_known_values() {
        assert!((cpu_relu_f32(0.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(-1.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(1.0) - 1.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(-0.001) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_relu_f32(100.0) - 100.0_f32).abs() < f32::EPSILON);
    }
    #[test]
    fn relu_precision_negative_zero() {
        assert!(cpu_relu_f32(-0.0) >= 0.0);
    }
    #[test]
    fn sigmoid_precision_known_values() {
        assert!((cpu_sigmoid_f32(0.0) - 0.5).abs() < 1e-7_f32);
        assert!((cpu_sigmoid_f32(100.0) - 1.0).abs() < 1e-6_f32);
        assert!(cpu_sigmoid_f32(-100.0).abs() < 1e-6_f32);
        let expected_sig1: f32 = 0.731_058_6;
        assert!(
            (cpu_sigmoid_f32(1.0) - expected_sig1).abs() < 1e-5_f32,
            "sigmoid(1.0) expected ~{expected_sig1}, got {}",
            cpu_sigmoid_f32(1.0)
        );
    }
    #[test]
    fn sigmoid_output_in_unit_interval() {
        let inputs: &[f32] = &[-10.0, -1.0, 0.0, 1.0, 10.0];
        for &x in inputs {
            let s = cpu_sigmoid_f32(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} not in (0,1)");
        }
        assert!(cpu_sigmoid_f32(-100.0) >= 0.0);
        assert!(cpu_sigmoid_f32(100.0) <= 1.0);
    }
    #[test]
    fn gelu_precision_known_values() {
        assert!(cpu_gelu_f32(0.0).abs() < 1e-7_f32);
        assert!(
            (cpu_gelu_f32(1.0) - 0.8413_f32).abs() < 0.001_f32,
            "gelu(1) should be ~0.8413, got {}",
            cpu_gelu_f32(1.0)
        );
        assert!(
            (cpu_gelu_f32(-1.0) + 0.1587_f32).abs() < 0.001_f32,
            "gelu(-1) should be ~-0.1587, got {}",
            cpu_gelu_f32(-1.0)
        );
        assert!(
            (cpu_gelu_f32(5.0) - 5.0_f32).abs() < 0.001_f32,
            "gelu(5) should be ~5.0, got {}",
            cpu_gelu_f32(5.0)
        );
    }
    #[test]
    fn gelu_sign_preservation() {
        assert!(cpu_gelu_f32(0.5) > 0.0);
        assert!(cpu_gelu_f32(2.0) > 0.0);
        assert!(cpu_gelu_f32(-2.0) < 0.0);
    }
    #[test]
    fn tanh_precision_known_values() {
        assert!(cpu_tanh_f32(0.0).abs() < 1e-7_f32);
        let expected_tanh1: f32 = 0.761_594_2;
        assert!(
            (cpu_tanh_f32(1.0) - expected_tanh1).abs() < 1e-5_f32,
            "tanh(1.0) expected ~{expected_tanh1}, got {}",
            cpu_tanh_f32(1.0)
        );
        assert!(
            (cpu_tanh_f32(-1.0) + expected_tanh1).abs() < 1e-5_f32,
            "tanh(-1.0) expected ~-{expected_tanh1}, got {}",
            cpu_tanh_f32(-1.0)
        );
        assert!(
            (cpu_tanh_f32(10.0) - 1.0).abs() < 1e-5_f32,
            "tanh(10) should be ~1.0"
        );
        assert!(
            (cpu_tanh_f32(-10.0) + 1.0).abs() < 1e-5_f32,
            "tanh(-10) should be ~-1.0"
        );
    }
    #[test]
    fn tanh_output_in_bounded_range() {
        let inputs: &[f32] = &[-5.0, -1.0, 0.0, 1.0, 5.0];
        for &x in inputs {
            let t = cpu_tanh_f32(x);
            assert!(t > -1.0 && t < 1.0, "tanh({x}) = {t} not in (-1,1)");
        }
        assert!(cpu_tanh_f32(-100.0) >= -1.0);
        assert!(cpu_tanh_f32(100.0) <= 1.0);
    }
    #[test]
    fn silu_precision_known_values() {
        assert!(cpu_silu_f32(0.0).abs() < 1e-7_f32);
        let expected_sig1: f32 = 0.731_058_6;
        assert!(
            (cpu_silu_f32(1.0) - expected_sig1).abs() < 1e-5_f32,
            "silu(1.0) expected ~{expected_sig1}, got {}",
            cpu_silu_f32(1.0)
        );
        assert!(
            (cpu_silu_f32(-1.0) + 0.2689_f32).abs() < 0.001_f32,
            "silu(-1) should be ~-0.2689, got {}",
            cpu_silu_f32(-1.0)
        );
    }
    #[test]
    fn silu_sign_matches_input() {
        for &x in &[0.1_f32, 0.5, 1.0, 2.0, 5.0] {
            assert!(
                cpu_silu_f32(x) > 0.0,
                "silu({x}) should be positive, got {}",
                cpu_silu_f32(x)
            );
        }
        for &x in &[-0.1_f32, -0.5, -2.0] {
            assert!(
                cpu_silu_f32(x) < 0.0,
                "silu({x}) should be negative, got {}",
                cpu_silu_f32(x)
            );
        }
    }
    #[test]
    fn elementwise_ptx_generates_fused_add_relu() {
        let tmpl =
            ElementwiseTemplate::new(ElementwiseOp::FusedAddRelu, PtxType::F32, SmVersion::Sm80);
        let ptx = tmpl
            .generate()
            .expect("FusedAddRelu should generate successfully");
        assert!(
            ptx.contains("add"),
            "fused kernel should contain add instruction"
        );
        assert!(
            ptx.contains("max"),
            "fused kernel should contain max for relu"
        );
    }
    #[test]
    fn elementwise_ops_precision_sweep() {
        let test_inputs: &[f32] = &[-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &x in test_inputs {
            assert!(
                cpu_relu_f32(x) >= 0.0,
                "relu({x}) = {} should be non-negative",
                cpu_relu_f32(x)
            );
            let s = cpu_sigmoid_f32(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s} should be in (0,1)");
            let t = cpu_tanh_f32(x);
            assert!(
                (-1.0_f32..=1.0).contains(&t),
                "tanh({x}) = {t} should be in [-1,1]"
            );
            if x > 0.1 {
                assert!(
                    cpu_silu_f32(x) > 0.0,
                    "silu({x}) should be positive for positive input"
                );
            }
        }
    }
    #[test]
    fn all_activation_ops_generate_ptx_for_f32() {
        let activation_ops = [
            ElementwiseOp::Relu,
            ElementwiseOp::Gelu,
            ElementwiseOp::Sigmoid,
            ElementwiseOp::Silu,
            ElementwiseOp::Tanh,
        ];
        for op in activation_ops {
            let t = ElementwiseTemplate::new(op, PtxType::F32, SmVersion::Sm80);
            let result = t.generate();
            assert!(
                result.is_ok(),
                "PTX generation failed for op {:?}: {:?}",
                op,
                result.err()
            );
            let ptx = result.expect("already checked is_ok");
            let name = op.as_str();
            assert!(
                ptx.contains(&format!(".entry elementwise_{name}_f32")),
                "PTX for {name} missing expected entry point"
            );
        }
    }
    #[test]
    fn relu_ptx_uses_max_instruction() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Relu, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("relu PTX generation should succeed");
        assert!(
            ptx.contains("max.f32"),
            "relu PTX must use max.f32 instruction"
        );
    }
    #[test]
    fn tanh_ptx_uses_tanh_or_approx_sequence() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Tanh, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("tanh PTX generation should succeed");
        let has_approx = ptx.contains("ex2.approx") || ptx.contains("tanh.approx");
        assert!(
            has_approx,
            "tanh PTX should use ex2.approx or tanh.approx, got:\n{ptx}"
        );
    }
    #[test]
    fn generate_one_minus_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OneMinus, PtxType::F32, SmVersion::Sm80);
        let ptx = t
            .generate()
            .expect("one_minus PTX generation should succeed");
        assert!(ptx.contains("sub.f32"), "one_minus must contain sub.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "one_minus must contain the 1.0 literal"
        );
        assert!(
            ptx.contains(".entry elementwise_one_minus_f32"),
            "one_minus must have correct kernel name"
        );
    }
    #[test]
    fn generate_pow_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Pow, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("pow PTX generation should succeed");
        assert!(
            ptx.contains("lg2.approx.f32"),
            "pow must contain lg2.approx.f32"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "pow must contain ex2.approx.f32"
        );
        assert!(ptx.contains("mul.f32"), "pow must contain mul.f32");
    }
    #[test]
    fn generate_min_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Min, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("min PTX generation should succeed");
        assert!(ptx.contains("min.f32"), "min must contain min.f32");
        assert!(
            ptx.contains(".entry elementwise_min_f32"),
            "min must have correct kernel name"
        );
    }
    #[test]
    fn generate_max_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Max, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("max PTX generation should succeed");
        assert!(ptx.contains("max.f32"), "max must contain max.f32");
        assert!(
            ptx.contains(".entry elementwise_max_f32"),
            "max must have correct kernel name"
        );
    }
    #[test]
    fn generate_cmp_eq_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::CmpEq, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("cmp_eq PTX generation should succeed");
        assert!(
            ptx.contains("setp.eq.f32"),
            "cmp_eq must contain setp.eq.f32"
        );
        assert!(ptx.contains("selp.f32"), "cmp_eq must contain selp.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "cmp_eq must contain the 1.0 literal for the true branch"
        );
    }
    #[test]
    fn generate_or_prob_sum_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OrProbSum, PtxType::F32, SmVersion::Sm80);
        let ptx = t
            .generate()
            .expect("or_prob_sum PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "or_prob_sum must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "or_prob_sum must contain sub.f32");
        assert!(ptx.contains("add.f32"), "or_prob_sum must contain add.f32");
    }
    #[test]
    fn generate_nand_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Nand, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("nand PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "nand must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "nand must contain sub.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "nand must contain the 1.0 literal"
        );
    }
    #[test]
    fn generate_nor_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Nor, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("nor PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "nor must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "nor must contain sub.f32");
        assert!(ptx.contains("add.f32"), "nor must contain add.f32");
        assert!(
            ptx.contains("0f3F800000"),
            "nor must contain the 1.0 literal"
        );
    }
    #[test]
    fn generate_xor_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::Xor, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("xor PTX generation should succeed");
        assert!(ptx.contains("mul.f32"), "xor must contain mul.f32");
        assert!(ptx.contains("sub.f32"), "xor must contain sub.f32");
        assert!(ptx.contains("add.f32"), "xor must contain add.f32");
        assert!(
            ptx.contains("0f40000000"),
            "xor must contain the 2.0 literal"
        );
    }
    #[test]
    fn generate_or_max_f32() {
        let t = ElementwiseTemplate::new(ElementwiseOp::OrMax, PtxType::F32, SmVersion::Sm80);
        let ptx = t.generate().expect("or_max PTX generation should succeed");
        assert!(ptx.contains("max.f32"), "or_max must use max.f32");
        assert!(
            ptx.contains(".entry elementwise_or_max_f32"),
            "or_max must have correct kernel name"
        );
    }
    #[test]
    fn generate_cmp_ops_f32() {
        let cases = [
            (ElementwiseOp::CmpNe, "setp.ne.f32"),
            (ElementwiseOp::CmpLt, "setp.lt.f32"),
            (ElementwiseOp::CmpGt, "setp.gt.f32"),
            (ElementwiseOp::CmpLe, "setp.le.f32"),
            (ElementwiseOp::CmpGe, "setp.ge.f32"),
        ];
        for (op, expected_instr) in cases {
            let t = ElementwiseTemplate::new(op, PtxType::F32, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX gen failed for {op:?}: {e}"));
            assert!(
                ptx.contains(expected_instr),
                "{op:?} PTX must contain {expected_instr}"
            );
            assert!(ptx.contains("selp.f32"), "{op:?} PTX must contain selp.f32");
        }
    }
    #[test]
    fn test_elementwise_ptx_has_valid_headers_extended() {
        let ops_and_types = [
            (ElementwiseOp::OneMinus, PtxType::F32),
            (ElementwiseOp::Pow, PtxType::F32),
            (ElementwiseOp::Min, PtxType::F32),
            (ElementwiseOp::Max, PtxType::F32),
            (ElementwiseOp::CmpEq, PtxType::F32),
            (ElementwiseOp::OrProbSum, PtxType::F32),
            (ElementwiseOp::Nand, PtxType::F32),
            (ElementwiseOp::Nor, PtxType::F32),
            (ElementwiseOp::Xor, PtxType::F32),
            (ElementwiseOp::OrMax, PtxType::F32),
        ];
        for (op, ty) in ops_and_types {
            let t = ElementwiseTemplate::new(op, ty, SmVersion::Sm80);
            let ptx = t
                .generate()
                .unwrap_or_else(|e| panic!("PTX generation failed for {op:?}: {e}"));
            assert!(
                ptx.contains(".version"),
                "PTX for {op:?} must have .version header"
            );
            assert!(
                ptx.contains(".target"),
                "PTX for {op:?} must have .target header"
            );
            assert!(
                ptx.contains(".entry"),
                "PTX for {op:?} must have .entry directive"
            );
        }
    }
    fn cpu_one_minus_f32(x: f32) -> f32 {
        1.0 - x
    }
    fn cpu_pow_f32(a: f32, b: f32) -> f32 {
        a.powf(b)
    }
    #[allow(clippy::float_cmp)]
    fn cpu_cmp_eq_f32(a: f32, b: f32) -> f32 {
        if a == b { 1.0 } else { 0.0 }
    }
    fn cpu_or_prob_sum_f32(a: f32, b: f32) -> f32 {
        a.mul_add(-b, a + b)
    }
    fn cpu_nand_f32(a: f32, b: f32) -> f32 {
        a.mul_add(-b, 1.0)
    }
    fn cpu_nor_f32(a: f32, b: f32) -> f32 {
        1.0 - a.mul_add(-b, a + b)
    }
    fn cpu_xor_f32(a: f32, b: f32) -> f32 {
        (2.0_f32 * a).mul_add(-b, a + b)
    }
    #[test]
    fn cpu_one_minus_f32_precision() {
        assert!((cpu_one_minus_f32(0.0) - 1.0_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(1.0) - 0.0_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(0.5) - 0.5_f32).abs() < f32::EPSILON);
        assert!((cpu_one_minus_f32(-1.0) - 2.0_f32).abs() < f32::EPSILON);
    }
    #[test]
    fn cpu_pow_f32_precision() {
        assert!((cpu_pow_f32(2.0, 3.0) - 8.0_f32).abs() < 1e-5_f32);
        assert!((cpu_pow_f32(4.0, 0.5) - 2.0_f32).abs() < 1e-5_f32);
        assert!((cpu_pow_f32(1.0, 100.0) - 1.0_f32).abs() < 1e-5_f32);
    }
    #[test]
    fn cpu_cmp_eq_f32_precision() {
        assert!((cpu_cmp_eq_f32(1.0, 1.0) - 1.0).abs() < f32::EPSILON);
        assert!((cpu_cmp_eq_f32(1.0, 2.0) - 0.0).abs() < f32::EPSILON);
        assert!((cpu_cmp_eq_f32(0.0, 0.0) - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn cpu_or_prob_sum_f32_precision() {
        assert!((cpu_or_prob_sum_f32(1.0, 1.0) - 1.0).abs() < f32::EPSILON);
        assert!(cpu_or_prob_sum_f32(0.0, 0.0).abs() < f32::EPSILON);
        assert!((cpu_or_prob_sum_f32(0.5, 0.5) - 0.75).abs() < 1e-6_f32);
    }
    #[test]
    fn cpu_nand_f32_precision() {
        assert!(cpu_nand_f32(1.0, 1.0).abs() < f32::EPSILON);
        assert!((cpu_nand_f32(0.0, 1.0) - 1.0).abs() < f32::EPSILON);
        assert!((cpu_nand_f32(0.5, 0.5) - 0.75).abs() < 1e-6_f32);
    }
    #[test]
    fn cpu_nor_f32_precision() {
        assert!((cpu_nor_f32(0.0, 0.0) - 1.0).abs() < f32::EPSILON);
        assert!(cpu_nor_f32(1.0, 0.0).abs() < f32::EPSILON);
        assert!((cpu_nor_f32(0.5, 0.5) - 0.25).abs() < 1e-6_f32);
    }
    #[test]
    fn cpu_xor_f32_precision() {
        assert!(cpu_xor_f32(0.0, 0.0).abs() < f32::EPSILON);
        assert!(cpu_xor_f32(1.0, 1.0).abs() < f32::EPSILON);
        assert!((cpu_xor_f32(1.0, 0.0) - 1.0).abs() < f32::EPSILON);
        assert!((cpu_xor_f32(0.5, 0.5) - 0.5).abs() < 1e-6_f32);
    }
    #[test]
    fn ptx_template_generates_fill_f32() {
        let template = ElementwiseTemplate::new(ElementwiseOp::Fill, PtxType::F32, SmVersion::Sm80);
        let ptx = template.generate().expect("fill PTX generation failed");
        assert!(
            ptx.contains("st.global.f32"),
            "must contain store instruction"
        );
        assert!(ptx.contains("elementwise_fill_f32"), "wrong kernel name");
    }
}
