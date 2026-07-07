//! Additional tests for the PTX validator — SM compatibility and register
//! pressure checks introduced in the second implementation pass.

use std::fmt::Write as _;

use super::{
    IrErrorKind, ValidationError, validate_branch_labels, validate_ir_instructions, validate_ptx,
    validate_ptx_for_target,
};
use crate::arch::SmVersion;
use crate::ir::{Instruction, Operand, PtxType, Register};

// ---------------------------------------------------------------------------
// SM compatibility tests
// ---------------------------------------------------------------------------

#[test]
fn test_valid_ptx_passes_validation() {
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry test_kernel(.param .u32 %param_n) {\n\
               .reg .u32 %r<4>;\n\
               mov.u32 %r0, %tid.x;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    assert!(
        result.is_ok(),
        "expected no errors for valid sm_80 PTX, got: {:?}",
        result.errors
    );
}

#[test]
fn test_cp_async_requires_sm80() {
    // cp.async on sm_75 should trigger SmIncompatibleInstruction
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               cp.async.ca.shared.global [%r0], [%rd0], 4;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for cp.async on sm_75"
    );
    // Verify the error names the required SM correctly
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_80"),
        "expected required_sm=sm_80, got: {msg}"
    );
}

#[test]
fn test_wgmma_requires_sm90() {
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry k() {\n\
               wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for wgmma on sm_80"
    );
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_90"),
        "expected required_sm=sm_90, got: {msg}"
    );
}

#[test]
fn test_fp8_requires_sm89() {
    // Using .e4m3 type in an sm_80 kernel
    let ptx = ".version 7.0\n.target sm_80\n.address_size 64\n\
               .visible .entry k() {\n\
               cvt.rn.e4m3x2.f32 %r0, %f0, %f1;\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm80);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        !compat_errors.is_empty(),
        "expected SmIncompatibleInstruction for fp8 e4m3 on sm_80"
    );
    let msg = format!("{}", compat_errors[0]);
    assert!(
        msg.contains("sm_89"),
        "expected required_sm=sm_89, got: {msg}"
    );
}

#[test]
fn test_too_many_registers_error() {
    // Build a PTX with more than 255 distinct registers (%f0..%f259 = 260 distinct)
    let mut ptx_body = String::from(
        ".version 7.0\n.target sm_80\n.address_size 64\n\
         .visible .entry k() {\n\
         .reg .f32 %f<260>;\n",
    );
    for i in 0..260_usize {
        let _ = writeln!(ptx_body, "    mov.f32 %f{i}, 0f00000000;");
    }
    ptx_body.push_str("    ret;\n}\n");

    let result = validate_ptx_for_target(&ptx_body, SmVersion::Sm80);
    // More than 255 DISTINCT VIRTUAL registers is valid PTX — ptxas spills the
    // excess to local memory. It must be reported as a warning, never a hard
    // error (255 is a post-allocation *physical* limit, not a source limit).
    assert!(
        !result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::RegisterPressureExceeded { .. })),
        "excess virtual registers must not be a hard error, errors: {:?}",
        result.errors
    );
    assert!(
        result.warnings.iter().any(|w| w.contains("260")),
        "expected a register-pressure warning for 260 virtual registers, warnings: {:?}",
        result.warnings
    );
}

#[test]
fn test_unknown_target_warns() {
    // A `.target` we cannot recognize disables SM/shared-mem checks; that loss
    // of coverage must surface as a warning rather than pass silently.
    let ptx = ".version 9.9\n.target sm_999\n.address_size 64\n\
               .visible .entry k() { ret; }\n";
    let result = validate_ptx(ptx);
    assert!(
        result.warnings.iter().any(|w| w.contains("sm_999")),
        "expected an unrecognized-target warning, warnings: {:?}",
        result.warnings
    );
    // A recognized target must NOT produce that warning.
    let good = ".version 8.5\n.target sm_80\n.address_size 64\n\
                .visible .entry k() { ret; }\n";
    let good_result = validate_ptx(good);
    assert!(
        !good_result
            .warnings
            .iter()
            .any(|w| w.contains("unrecognized .target")),
        "recognized target should not warn, warnings: {:?}",
        good_result.warnings
    );
}

#[test]
fn test_shared_memory_over_limit_error() {
    // sm_75 allows 65536 bytes; declare 70000 bytes
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               .shared .align 4 .b8 smem[70000];\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    assert!(
        result.has_errors(),
        "expected error for shared memory exceeding sm_75 limit"
    );
    assert!(
        result.errors.iter().any(|e| matches!(
            e,
            ValidationError::InvalidSharedMemSize {
                declared: 70000,
                ..
            }
        )),
        "expected InvalidSharedMemSize error, got: {:?}",
        result.errors
    );
}

#[test]
fn test_wgmma_valid_on_sm90() {
    // wgmma on sm_90 should pass the SM compatibility check
    let ptx = ".version 8.0\n.target sm_90\n.address_size 64\n\
               .visible .entry k() {\n\
               wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm90);
    // Should not have any SM-compatibility errors
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e, ValidationError::SmIncompatibleInstruction { .. }))
        .collect();
    assert!(
        compat_errors.is_empty(),
        "wgmma should be valid on sm_90, got errors: {compat_errors:?}"
    );
}

#[test]
fn test_empty_ptx_valid() {
    // Empty PTX should report missing directives but no SM/register errors
    let result = validate_ptx("");
    // Should have MissingVersionDirective and MissingTargetDirective errors
    assert!(result.has_errors());
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingVersionDirective))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingTargetDirective))
    );
    // No SM or register pressure errors (nothing to scan)
    assert!(!result.errors.iter().any(|e| matches!(
        e,
        ValidationError::SmIncompatibleInstruction { .. }
            | ValidationError::RegisterPressureExceeded { .. }
    )));
}

#[test]
fn test_sm_compat_error_display() {
    let err = ValidationError::SmIncompatibleInstruction {
        instruction: "cp.async".to_string(),
        required_sm: "sm_80".to_string(),
        found_sm: "sm_75".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("cp.async"));
    assert!(msg.contains("sm_80"));
    assert!(msg.contains("sm_75"));
}

#[test]
fn test_register_pressure_exceeded_display() {
    let err = ValidationError::RegisterPressureExceeded {
        count: 300,
        max_allowed: 255,
    };
    let msg = format!("{err}");
    assert!(msg.contains("300"));
    assert!(msg.contains("255"));
}

#[test]
fn test_mma_sync_requires_sm75() {
    // mma.sync is available on sm_75+ — verify sm_75 target passes.
    let ptx = ".version 6.4\n.target sm_75\n.address_size 64\n\
               .visible .entry k() {\n\
               mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {...};\n\
               ret;\n\
               }\n";
    let result = validate_ptx_for_target(ptx, SmVersion::Sm75);
    let compat_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| {
            matches!(e, ValidationError::SmIncompatibleInstruction { instruction, .. }
                if instruction.contains("mma.sync"))
        })
        .collect();
    // sm_75 supports mma.sync, so no error expected
    assert!(
        compat_errors.is_empty(),
        "mma.sync should be valid on sm_75: {compat_errors:?}"
    );
}

// ---------------------------------------------------------------------------
// Branch / label validation tests (F085)
// ---------------------------------------------------------------------------

fn reg(name: &str, ty: PtxType) -> Register {
    Register {
        name: name.to_string(),
        ty,
    }
}

#[test]
fn test_branch_to_undefined_label_errors() {
    let instructions = vec![
        Instruction::Branch {
            target: "nowhere".to_string(),
            predicate: None,
        },
        Instruction::Label("elsewhere".to_string()),
        Instruction::Return,
    ];
    let result = validate_ir_instructions(&instructions);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::InvalidOperand && e.message.contains("nowhere")),
        "expected an undefined-branch-target error, got: {:?}",
        result.errors
    );
}

#[test]
fn test_branch_to_defined_label_ok() {
    let instructions = vec![
        Instruction::Branch {
            target: "done".to_string(),
            predicate: None,
        },
        Instruction::Label("done".to_string()),
        Instruction::Return,
    ];
    let result = validate_branch_labels(&instructions);
    assert!(
        result.errors.is_empty(),
        "forward branch to a defined label must be valid, got: {:?}",
        result.errors
    );
}

#[test]
fn test_duplicate_label_errors() {
    let instructions = vec![
        Instruction::Label("loop".to_string()),
        Instruction::Return,
        Instruction::Label("loop".to_string()),
    ];
    let result = validate_branch_labels(&instructions);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::InvalidOperand && e.message.contains("duplicate")),
        "expected a duplicate-label error, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// ldmatrix operand consistency (F106)
// ---------------------------------------------------------------------------

#[test]
fn test_ldmatrix_fragment_count_mismatch_errors() {
    // num_fragments = 4 but only 2 destination registers supplied.
    let instructions = vec![Instruction::Ldmatrix {
        num_fragments: 4,
        trans: false,
        dst_regs: vec![reg("%r0", PtxType::B32), reg("%r1", PtxType::B32)],
        src_addr: Operand::Register(reg("%rd0", PtxType::U64)),
    }];
    let result = validate_ir_instructions(&instructions);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::InvalidOperand && e.message.contains("ldmatrix")),
        "expected an ldmatrix fragment-count mismatch error, got: {:?}",
        result.errors
    );
}

#[test]
fn test_ldmatrix_invalid_fragment_count_errors() {
    // num_fragments = 3 is not a legal ldmatrix width.
    let instructions = vec![Instruction::Ldmatrix {
        num_fragments: 3,
        trans: false,
        dst_regs: vec![
            reg("%r0", PtxType::B32),
            reg("%r1", PtxType::B32),
            reg("%r2", PtxType::B32),
        ],
        src_addr: Operand::Register(reg("%rd0", PtxType::U64)),
    }];
    let result = validate_ir_instructions(&instructions);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.kind == IrErrorKind::InvalidOperand && e.message.contains("1, 2, or 4")),
        "expected an ldmatrix invalid-width error, got: {:?}",
        result.errors
    );
}

#[test]
fn test_ldmatrix_valid_x4_ok() {
    let instructions = vec![Instruction::Ldmatrix {
        num_fragments: 4,
        trans: false,
        dst_regs: vec![
            reg("%r0", PtxType::B32),
            reg("%r1", PtxType::B32),
            reg("%r2", PtxType::B32),
            reg("%r3", PtxType::B32),
        ],
        src_addr: Operand::Register(reg("%rd0", PtxType::U64)),
    }];
    // Only check tensor-core operand validation (no register-lifetime noise).
    let mut has_ldmatrix_err = false;
    for e in &validate_ir_instructions(&instructions).errors {
        if e.message.contains("ldmatrix") {
            has_ldmatrix_err = true;
        }
    }
    assert!(
        !has_ldmatrix_err,
        "valid x4 ldmatrix must not produce an ldmatrix operand error"
    );
}
