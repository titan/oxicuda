use super::*;
use crate::ir::{
    BasicBlock, CacheQualifier, CmpOp, ImmValue, Instruction, MemorySpace, Operand, PtxFunction,
    PtxModule, PtxType, Register, RoundingMode, SpecialReg, VectorWidth,
};

// -- Test helpers -------------------------------------------------------

fn make_reg(name: &str, ty: PtxType) -> Register {
    Register {
        name: name.to_string(),
        ty,
    }
}

fn make_operand_reg(name: &str, ty: PtxType) -> Operand {
    Operand::Register(make_reg(name, ty))
}

fn make_simple_function() -> PtxFunction {
    let mut func = PtxFunction::new("test_kernel");
    func.add_param("a_ptr", PtxType::U64);
    func.add_param("n", PtxType::U32);

    // Load param
    func.push(Instruction::LoadParam {
        ty: PtxType::U64,
        dst: make_reg("%rd0", PtxType::U64),
        param_name: "a_ptr".to_string(),
    });

    // MovSpecial
    func.push(Instruction::MovSpecial {
        dst: make_reg("%r0", PtxType::U32),
        special: SpecialReg::TidX,
    });

    // Add
    func.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r1", PtxType::U32),
        a: make_operand_reg("%r0", PtxType::U32),
        b: Operand::Immediate(ImmValue::U32(1)),
    });

    // Load
    func.push(Instruction::Load {
        space: MemorySpace::Global,
        qualifier: CacheQualifier::None,
        vec: VectorWidth::V1,
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        addr: Operand::Address {
            base: make_reg("%rd0", PtxType::U64),
            offset: None,
        },
    });

    // Fma
    func.push(Instruction::Fma {
        rnd: RoundingMode::Rn,
        ty: PtxType::F32,
        dst: make_reg("%f1", PtxType::F32),
        a: make_operand_reg("%f0", PtxType::F32),
        b: Operand::Immediate(ImmValue::F32(2.0)),
        c: Operand::Immediate(ImmValue::F32(1.0)),
    });

    // Store
    func.push(Instruction::Store {
        space: MemorySpace::Global,
        qualifier: CacheQualifier::None,
        vec: VectorWidth::V1,
        ty: PtxType::F32,
        addr: Operand::Address {
            base: make_reg("%rd0", PtxType::U64),
            offset: None,
        },
        src: make_reg("%f1", PtxType::F32),
    });

    func.push(Instruction::Return);
    func
}

fn make_branching_function() -> PtxFunction {
    let mut func = PtxFunction::new("branch_kernel");

    func.push(Instruction::MovSpecial {
        dst: make_reg("%r0", PtxType::U32),
        special: SpecialReg::TidX,
    });

    func.push(Instruction::SetP {
        cmp: CmpOp::Lt,
        ty: PtxType::U32,
        dst: make_reg("%p0", PtxType::Pred),
        a: make_operand_reg("%r0", PtxType::U32),
        b: Operand::Immediate(ImmValue::U32(128)),
    });

    func.push(Instruction::Branch {
        target: "skip".to_string(),
        predicate: Some((make_reg("%p0", PtxType::Pred), true)),
    });

    // Work block
    func.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r1", PtxType::U32),
        a: make_operand_reg("%r0", PtxType::U32),
        b: Operand::Immediate(ImmValue::U32(1)),
    });

    func.push(Instruction::Label("skip".to_string()));

    func.push(Instruction::Return);
    func
}

// -- Tests --------------------------------------------------------------

#[test]
fn test_render_empty_function() {
    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let func = PtxFunction::new("empty");
    let output = explorer.render_function(&func);
    assert!(output.contains("empty"));
    assert!(output.contains('{'));
    assert!(output.contains('}'));
}

#[test]
fn test_render_function_with_multiple_blocks() {
    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let func = make_branching_function();
    let output = explorer.render_function(&func);
    assert!(output.contains("branch_kernel"));
    assert!(output.contains("setp"));
    assert!(output.contains("bra"));
    assert!(output.contains("add"));
}

#[test]
fn test_cfg_rendering_with_branches() {
    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let func = make_branching_function();
    let output = explorer.render_cfg(&func);
    assert!(output.contains("Control Flow Graph"));
    assert!(output.contains("skip"));
    assert!(output.contains("-->"));
}

#[test]
fn test_register_lifetime_analysis() {
    let analyzer = RegisterLifetimeAnalyzer;
    let func = make_simple_function();
    let lifetimes = analyzer.analyze(&func);

    // We should have several registers
    assert!(!lifetimes.is_empty());

    // %rd0 is defined first (LoadParam at index 0) and used later (Load, Store)
    let rd0 = lifetimes.iter().find(|l| l.register == "%rd0");
    assert!(rd0.is_some(), "should find %rd0 lifetime");
    let rd0 = rd0.expect("checked above");
    assert_eq!(rd0.first_def, 0);
    assert!(
        rd0.last_use > rd0.first_def,
        "last_use should be after first_def"
    );
}

#[test]
fn test_register_lifetime_timeline_rendering() {
    let analyzer = RegisterLifetimeAnalyzer;
    let func = make_simple_function();
    let lifetimes = analyzer.analyze(&func);
    let timeline = RegisterLifetimeAnalyzer::render_timeline(&lifetimes, 80);
    assert!(timeline.contains("Register Lifetimes"));
    assert!(timeline.contains('#')); // should have bar characters
    assert!(timeline.contains("uses:"));
}

#[test]
fn test_instruction_mix_categorization() {
    let analyzer = InstructionMixAnalyzer;
    let func = make_simple_function();
    let mix = analyzer.analyze(&func);

    assert_eq!(mix.total, func.body.len());

    // Should have arithmetic, memory, special, and control categories
    let arith = mix
        .counts
        .get(&InstructionCategory::Arithmetic)
        .copied()
        .unwrap_or(0);
    let mem = mix
        .counts
        .get(&InstructionCategory::Memory)
        .copied()
        .unwrap_or(0);
    let special = mix
        .counts
        .get(&InstructionCategory::Special)
        .copied()
        .unwrap_or(0);
    assert!(arith > 0, "should have arithmetic instructions");
    assert!(mem > 0, "should have memory instructions");
    assert!(special > 0, "should have special instructions");
}

#[test]
fn test_instruction_mix_bar_chart() {
    let analyzer = InstructionMixAnalyzer;
    let func = make_simple_function();
    let mix = analyzer.analyze(&func);
    let chart = InstructionMixAnalyzer::render_bar_chart(&mix, 80);
    assert!(chart.contains("Instruction Mix"));
    assert!(chart.contains('#')); // bar characters
    assert!(chart.contains('%')); // percentages
    assert!(chart.contains("Total:"));
}

#[test]
fn test_memory_access_pattern_analysis() {
    let func = make_simple_function();
    let report = MemoryAccessPattern::analyze(&func);
    assert_eq!(report.global_loads, 1);
    assert_eq!(report.global_stores, 1);
    assert_eq!(report.shared_loads, 0);
    assert_eq!(report.shared_stores, 0);
    assert!(report.coalescing_score > 0.0);
    assert!(report.coalescing_score <= 1.0);
}

#[test]
fn test_ptx_diff_identical_functions() {
    let func = make_simple_function();
    let report = PtxDiff::diff(&func, &func);
    assert_eq!(report.added_instructions, 0);
    assert_eq!(report.removed_instructions, 0);
    assert_eq!(report.changed_blocks, 0);
    assert_eq!(report.register_delta, 0);
}

#[test]
fn test_ptx_diff_different_functions() {
    let a = make_simple_function();
    let mut b = make_simple_function();
    // Add extra instructions to b
    b.push(Instruction::Comment("extra".to_string()));
    b.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r99", PtxType::U32),
        a: Operand::Immediate(ImmValue::U32(0)),
        b: Operand::Immediate(ImmValue::U32(1)),
    });

    let report = PtxDiff::diff(&a, &b);
    assert!(report.added_instructions > 0);
    assert!(report.register_delta > 0);

    let rendered = PtxDiff::render_diff(&report);
    assert!(rendered.contains("PTX Diff Report"));
    assert!(rendered.contains('+'));
}

#[test]
fn test_kernel_complexity_scoring() {
    let func = make_branching_function();
    let metrics = KernelComplexityScore::analyze(&func);
    assert_eq!(metrics.instruction_count, func.body.len());
    assert!(metrics.branch_count > 0, "should detect branches");
    assert!(metrics.estimated_occupancy_pct > 0.0);
    assert!(metrics.estimated_occupancy_pct <= 100.0);
}

#[test]
fn test_color_vs_no_color_output() {
    let func = make_simple_function();

    let no_color = PtxExplorer::new(ExplorerConfig {
        use_color: false,
        ..ExplorerConfig::default()
    });
    let with_color = PtxExplorer::new(ExplorerConfig {
        use_color: true,
        ..ExplorerConfig::default()
    });

    let plain = no_color.render_function(&func);
    let colored = with_color.render_function(&func);

    // Colored output should contain ANSI codes
    assert!(colored.contains("\x1b["));
    // Plain output should not
    assert!(!plain.contains("\x1b["));
    // Both should contain the function name
    assert!(plain.contains("test_kernel"));
    assert!(colored.contains("test_kernel"));
}

#[test]
fn test_config_defaults() {
    let config = ExplorerConfig::default();
    assert!(!config.use_color);
    assert_eq!(config.max_width, 120);
    assert!(!config.show_line_numbers);
    assert!(!config.show_register_types);
    assert!(!config.show_instruction_latency);
}

#[test]
fn test_large_function_handling() {
    let mut func = PtxFunction::new("big_kernel");
    // Add 500 instructions -- should not crash or truncate
    for i in 0..500 {
        func.push(Instruction::Add {
            ty: PtxType::F32,
            dst: make_reg(&format!("%f{i}"), PtxType::F32),
            a: Operand::Immediate(ImmValue::F32(1.0)),
            b: Operand::Immediate(ImmValue::F32(2.0)),
        });
    }

    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let output = explorer.render_function(&func);
    // Should contain all 500 instructions
    assert!(output.lines().count() > 500);

    let mix = InstructionMixAnalyzer.analyze(&func);
    assert_eq!(mix.total, 500);

    let metrics = KernelComplexityScore::analyze(&func);
    assert_eq!(metrics.instruction_count, 500);
}

#[test]
fn test_line_number_rendering() {
    let config = ExplorerConfig {
        show_line_numbers: true,
        ..ExplorerConfig::default()
    };
    let explorer = PtxExplorer::new(config);
    let func = make_simple_function();
    let output = explorer.render_function(&func);
    // Should contain line numbers starting from 1
    assert!(output.contains("   1  "));
    assert!(output.contains("   2  "));
}

#[test]
fn test_render_module() {
    let mut module = PtxModule::new("sm_80");
    module.add_function(make_simple_function());
    module.add_function(make_branching_function());

    let explorer = PtxExplorer::new(ExplorerConfig::default());
    let output = explorer.render_module(&module);
    assert!(output.contains(".version 8.5"));
    assert!(output.contains(".target sm_80"));
    assert!(output.contains("test_kernel"));
    assert!(output.contains("branch_kernel"));
}

#[test]
fn test_dependency_graph() {
    let mut block = BasicBlock::with_label("test_block");
    block.push(Instruction::LoadParam {
        ty: PtxType::F32,
        dst: make_reg("%f0", PtxType::F32),
        param_name: "x".to_string(),
    });
    block.push(Instruction::Add {
        ty: PtxType::F32,
        dst: make_reg("%f1", PtxType::F32),
        a: make_operand_reg("%f0", PtxType::F32),
        b: Operand::Immediate(ImmValue::F32(1.0)),
    });
    block.push(Instruction::Add {
        ty: PtxType::F32,
        dst: make_reg("%f2", PtxType::F32),
        a: make_operand_reg("%f1", PtxType::F32),
        b: make_operand_reg("%f0", PtxType::F32),
    });

    let explorer = PtxExplorer::new(ExplorerConfig::default());
    let output = explorer.render_dependency_graph(&block);
    assert!(output.contains("Dependency graph"));
    assert!(output.contains("test_block"));
    assert!(output.contains("-->")); // should have dependency edges
    assert!(output.contains("%f0")); // register dependency
}

#[test]
fn test_cfg_empty_function() {
    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let func = PtxFunction::new("empty_kernel");
    let output = explorer.render_cfg(&func);
    // Empty function body → empty CFG message
    assert!(
        output.contains("empty CFG")
            || output.contains("Control Flow Graph")
            || output.is_empty()
            || output.contains("(entry)")
    );
}

#[test]
fn test_cfg_no_branch_single_block() {
    let config = ExplorerConfig::default();
    let explorer = PtxExplorer::new(config);
    let func = make_simple_function();
    let output = explorer.render_cfg(&func);
    // make_simple_function has no labels/branches → one block
    assert!(output.contains("Control Flow Graph"));
    assert!(output.contains("B0"));
}

#[test]
fn test_register_lifetime_single_instruction() {
    let analyzer = RegisterLifetimeAnalyzer;
    let mut func = PtxFunction::new("single");
    func.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        a: Operand::Immediate(ImmValue::U32(1)),
        b: Operand::Immediate(ImmValue::U32(2)),
    });
    let lifetimes = analyzer.analyze(&func);
    // %r0 is written at index 0, last_use = 0
    let r0 = lifetimes.iter().find(|l| l.register == "%r0");
    assert!(r0.is_some(), "should track %r0");
    let r0 = r0.expect("checked above");
    assert_eq!(r0.first_def, 0);
    assert_eq!(r0.last_use, 0);
}

#[test]
fn test_register_lifetime_render_empty() {
    let rendered = RegisterLifetimeAnalyzer::render_timeline(&[], 80);
    assert!(rendered.contains("no registers"));
}

#[test]
fn test_instruction_mix_empty_function() {
    let analyzer = InstructionMixAnalyzer;
    let func = PtxFunction::new("empty_kernel");
    let mix = analyzer.analyze(&func);
    assert_eq!(mix.total, 0);
    let chart = InstructionMixAnalyzer::render_bar_chart(&mix, 80);
    assert!(chart.contains("no instructions"));
}

#[test]
fn test_dependency_graph_no_deps() {
    let mut block = BasicBlock::with_label("no_deps");
    // Instructions with no register dependencies between them
    block.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r0", PtxType::U32),
        a: Operand::Immediate(ImmValue::U32(1)),
        b: Operand::Immediate(ImmValue::U32(2)),
    });
    block.push(Instruction::Add {
        ty: PtxType::U32,
        dst: make_reg("%r1", PtxType::U32),
        a: Operand::Immediate(ImmValue::U32(3)),
        b: Operand::Immediate(ImmValue::U32(4)),
    });
    let explorer = PtxExplorer::new(ExplorerConfig::default());
    let output = explorer.render_dependency_graph(&block);
    assert!(output.contains("no_deps"));
    // %r0 and %r1 are independent — no edges between them
    assert!(output.contains("no data dependencies"));
}

#[test]
fn test_cfg_renderer_empty_blocks() {
    let renderer = CfgRenderer;
    let output = renderer.render(&[]);
    assert!(output.contains("empty CFG"));
}
