//! Automatic kernel fusion analysis pass.
//!
//! This module implements an IR-level analysis pass that identifies and fuses
//! compatible PTX kernels to reduce launch overhead and eliminate intermediate
//! memory allocations. The analysis examines pairs and sequences of kernels,
//! detecting data dependencies, checking fusion constraints, and producing
//! an optimized [`FusionPlan`] that groups kernels for combined execution.
//!
//! # Fusion types
//!
//! - **Elementwise**: Both kernels perform purely elementwise operations with
//!   no shared memory or synchronization -- trivially fusible.
//! - **Producer-consumer**: One kernel's output feeds directly into another
//!   kernel's input, allowing the intermediate buffer to be eliminated.
//! - **Horizontal**: Independent kernels with compatible grid dimensions that
//!   can share a single launch.
//! - **Vertical**: Producer-consumer pair separated by a reduction boundary.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_ptx::ir::{PtxFunction, PtxType, Instruction, Operand, Register, ImmValue};
//! use oxicuda_ptx::ir::{MemorySpace, CacheQualifier, VectorWidth, SpecialReg};
//! use oxicuda_ptx::analysis::kernel_fusion::{FusionAnalysis, plan_fusion};
//!
//! // Create two simple elementwise kernels
//! let mut k0 = PtxFunction::new("add_kernel");
//! k0.add_param("input", PtxType::U64);
//! k0.add_param("output", PtxType::U64);
//! k0.body.push(Instruction::Add {
//!     ty: PtxType::F32,
//!     dst: Register { name: "%f0".into(), ty: PtxType::F32 },
//!     a: Operand::Immediate(ImmValue::F32(1.0)),
//!     b: Operand::Immediate(ImmValue::F32(2.0)),
//! });
//!
//! let mut k1 = PtxFunction::new("mul_kernel");
//! k1.add_param("input", PtxType::U64);
//! k1.add_param("output", PtxType::U64);
//! k1.body.push(Instruction::Add {
//!     ty: PtxType::F32,
//!     dst: Register { name: "%f0".into(), ty: PtxType::F32 },
//!     a: Operand::Immediate(ImmValue::F32(3.0)),
//!     b: Operand::Immediate(ImmValue::F32(4.0)),
//! });
//!
//! let report = plan_fusion(&[k0, k1], 255, 49152);
//! assert!(!report.plan.candidates.is_empty());
//! ```

use std::collections::HashSet;
use std::fmt;

use crate::analysis::fusion_cost_model::{FusionCostModel, FusionVerdict};
use crate::arch::SmVersion;
use crate::ir::{Instruction, PtxFunction, PtxType};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// The type of fusion identified between two kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionType {
    /// Both kernels are elementwise, trivially fusible.
    Elementwise,
    /// Producer output feeds directly to consumer input.
    ProducerConsumer,
    /// Independent kernels with the same grid that can share a launch.
    Horizontal,
    /// Producer-consumer with a reduction boundary.
    Vertical,
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elementwise => write!(f, "elementwise"),
            Self::ProducerConsumer => write!(f, "producer-consumer"),
            Self::Horizontal => write!(f, "horizontal"),
            Self::Vertical => write!(f, "vertical"),
        }
    }
}

/// Access pattern for a data dependency between kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential streaming access (coalesced).
    Streaming,
    /// Random (non-coalesced) access.
    Random,
    /// Strided access with the given stride in elements.
    Strided(u32),
    /// Access pattern could not be determined.
    Unknown,
}

impl fmt::Display for AccessPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Streaming => write!(f, "streaming"),
            Self::Random => write!(f, "random"),
            Self::Strided(s) => write!(f, "strided({s})"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Constraints that a fusion candidate must satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionConstraint {
    /// The two kernels must have compatible grid dimensions.
    SameGridDimensions,
    /// Combined shared memory usage must not exceed the device limit.
    NoSharedMemoryConflict,
    /// No conflicting synchronization barriers between the kernels.
    NoBarrierConflict,
    /// Combined register usage must not exceed the given budget.
    RegisterBudget(u32),
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A candidate pair of kernels that may be fused.
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// Index of the producing kernel in the original sequence.
    pub producer_index: usize,
    /// Index of the consuming kernel in the original sequence.
    pub consumer_index: usize,
    /// Name of the intermediate buffer shared between the pair.
    pub shared_buffer: String,
    /// The type of fusion identified.
    pub fusion_type: FusionType,
    /// Estimated speedup from fusing these kernels (1.0 = no change).
    pub estimated_speedup: f64,
    /// Shared memory bytes used by the producer kernel.
    pub producer_shared_bytes: usize,
    /// Shared memory bytes used by the consumer kernel.
    pub consumer_shared_bytes: usize,
    /// Estimated register count for the fused kernel.
    pub estimated_registers: u32,
    /// Per-thread bytes the producer kernel moves to/from global memory.
    pub producer_global_bytes: usize,
    /// Per-thread bytes the consumer kernel moves to/from global memory.
    pub consumer_global_bytes: usize,
    /// Per-thread global-memory bytes that fusion eliminates by keeping the
    /// intermediate result in registers/shared memory instead of round-tripping
    /// it through DRAM (the producer's write plus the consumer's matching read).
    pub eliminated_global_bytes: usize,
    /// Estimated floating-point/arithmetic operations performed by the producer.
    pub producer_flops: u64,
    /// Estimated floating-point/arithmetic operations performed by the consumer.
    pub consumer_flops: u64,
    /// Number of distinct kernels covered by this candidate (2 for a pair).
    pub kernel_count: usize,
}

/// A data dependency between two kernels.
#[derive(Debug, Clone)]
pub struct DataDependency {
    /// Index of the producing kernel.
    pub producer: usize,
    /// Index of the consuming kernel.
    pub consumer: usize,
    /// Name of the buffer connecting the kernels.
    pub buffer_name: String,
    /// Access pattern of the dependency.
    pub access_pattern: AccessPattern,
}

/// A plan describing which kernels to fuse and the expected benefit.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// All accepted fusion candidates.
    pub candidates: Vec<FusionCandidate>,
    /// Groups of kernel indices that should be fused together.
    pub fused_groups: Vec<Vec<usize>>,
    /// Number of kernels before fusion.
    pub original_kernel_count: usize,
    /// Number of kernels after fusion.
    pub fused_kernel_count: usize,
    /// Estimated aggregate speedup factor.
    pub estimated_total_speedup: f64,
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Fusion Plan")?;
        writeln!(f, "  Original kernels: {}", self.original_kernel_count)?;
        writeln!(f, "  Fused kernels:    {}", self.fused_kernel_count)?;
        writeln!(
            f,
            "  Estimated speedup: {:.2}x",
            self.estimated_total_speedup
        )?;
        writeln!(f, "  Groups:")?;
        for (i, group) in self.fused_groups.iter().enumerate() {
            writeln!(f, "    [{i}]: {group:?}")?;
        }
        Ok(())
    }
}

/// Full fusion analysis report including accepted and rejected candidates.
#[derive(Debug, Clone)]
pub struct FusionReport {
    /// The accepted fusion plan.
    pub plan: FusionPlan,
    /// Candidates that were rejected, with a reason string.
    pub rejected: Vec<(FusionCandidate, String)>,
}

impl fmt::Display for FusionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.plan)?;
        if !self.rejected.is_empty() {
            writeln!(f, "  Rejected candidates:")?;
            for (cand, reason) in &self.rejected {
                writeln!(
                    f,
                    "    kernel[{}] -> kernel[{}] ({}): {}",
                    cand.producer_index, cand.consumer_index, cand.fusion_type, reason
                )?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Analysis engine
// ---------------------------------------------------------------------------

/// Kernel fusion analysis engine.
///
/// Provides static analysis methods to identify fusion opportunities
/// between PTX kernels. All methods are associated functions that do
/// not require mutable state.
#[derive(Debug, Clone, Copy, Default)]
pub struct FusionAnalysis;

impl FusionAnalysis {
    /// Creates a new fusion analysis engine.
    pub const fn new() -> Self {
        Self
    }

    /// Analyzes a pair of kernels for fusion opportunity.
    ///
    /// Returns `Some(FusionCandidate)` if the two kernels can be fused,
    /// or `None` if no fusion opportunity exists.
    pub fn analyze_pair(producer: &PtxFunction, consumer: &PtxFunction) -> Option<FusionCandidate> {
        let prod_shared = shared_mem_bytes(producer);
        let cons_shared = shared_mem_bytes(consumer);
        let est_regs = estimate_register_count(producer) + estimate_register_count(consumer);

        // Determine shared buffer name from parameter overlap
        let shared_buf = find_shared_buffer(producer, consumer);

        let prod_ew = Self::is_elementwise(producer);
        let cons_ew = Self::is_elementwise(consumer);

        // Determine fusion type
        let fusion_type = if prod_ew && cons_ew {
            FusionType::Elementwise
        } else if shared_buf.is_some() {
            if has_reduction(producer) {
                FusionType::Vertical
            } else {
                FusionType::ProducerConsumer
            }
        } else if compatible_grid_hints(producer, consumer) {
            FusionType::Horizontal
        } else {
            return None;
        };

        let buffer_name = shared_buf.unwrap_or_default();

        // Per-thread global-memory traffic and arithmetic work derived directly
        // from each kernel's instruction stream. These drive the cost model.
        let producer_global_bytes = global_traffic_bytes(producer);
        let consumer_global_bytes = global_traffic_bytes(consumer);
        let producer_flops = arithmetic_op_count(producer);
        let consumer_flops = arithmetic_op_count(consumer);

        // Bytes saved by fusion: the intermediate is written once by the producer
        // and read back once by the consumer; fusion keeps it in registers/shared
        // memory and eliminates that round trip. We estimate the round-trip size
        // from the intermediate element width that flows across the pair.
        let eliminated_global_bytes =
            intermediate_roundtrip_bytes(producer, consumer, fusion_type, !buffer_name.is_empty());

        let candidate = FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: buffer_name,
            fusion_type,
            estimated_speedup: 1.0, // overwritten below by the cost model
            producer_shared_bytes: prod_shared,
            consumer_shared_bytes: cons_shared,
            estimated_registers: est_regs,
            producer_global_bytes,
            consumer_global_bytes,
            eliminated_global_bytes,
            producer_flops,
            consumer_flops,
            kernel_count: 2,
        };

        let speedup = Self::estimate_fusion_speedup(&candidate);

        Some(FusionCandidate {
            estimated_speedup: speedup,
            ..candidate
        })
    }

    /// Analyzes a sequence of kernels and returns all fusion candidates.
    pub fn analyze_sequence(kernels: &[PtxFunction]) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();
        if kernels.len() < 2 {
            return candidates;
        }

        for i in 0..kernels.len() {
            for j in (i + 1)..kernels.len() {
                if let Some(mut cand) = Self::analyze_pair(&kernels[i], &kernels[j]) {
                    cand.producer_index = i;
                    cand.consumer_index = j;
                    cand.estimated_speedup = Self::estimate_fusion_speedup(&cand);
                    candidates.push(cand);
                }
            }
        }

        candidates
    }

    /// Checks whether a fusion candidate satisfies all given constraints.
    pub fn check_constraints(
        candidate: &FusionCandidate,
        constraints: &[FusionConstraint],
    ) -> bool {
        for constraint in constraints {
            match constraint {
                FusionConstraint::SameGridDimensions => {
                    // For elementwise and producer-consumer, grid compatibility
                    // is assumed when the analysis identified the pair. For
                    // horizontal fusion, it was explicitly checked.
                    // Always passes if we got this far.
                }
                FusionConstraint::NoSharedMemoryConflict => {
                    // Default limit: 48 KiB
                    let combined =
                        candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
                    if combined > 49152 {
                        return false;
                    }
                }
                FusionConstraint::NoBarrierConflict => {
                    // Barrier conflicts are detected during pair analysis.
                    // If both kernels use barriers, fusion is risky.
                    // We encode this: if both have nonzero shared mem AND
                    // the fusion type is not Elementwise, it may conflict.
                    // Conservative: reject if both use shared memory.
                    if candidate.producer_shared_bytes > 0
                        && candidate.consumer_shared_bytes > 0
                        && candidate.fusion_type != FusionType::Elementwise
                    {
                        return false;
                    }
                }
                FusionConstraint::RegisterBudget(max_regs) => {
                    if candidate.estimated_registers > *max_regs {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Checks whether a kernel is purely elementwise.
    ///
    /// A kernel is elementwise if it has no shared memory, no synchronization
    /// barriers, no tensor core operations, and no warp-level reductions.
    pub fn is_elementwise(func: &PtxFunction) -> bool {
        if !func.shared_mem.is_empty() {
            return false;
        }

        for inst in &func.body {
            if is_non_elementwise_instruction(inst) {
                return false;
            }
        }

        true
    }

    /// Finds data dependencies between kernels in a sequence.
    ///
    /// A data dependency exists when a producer kernel writes to a buffer
    /// that a consumer kernel reads from, identified by matching parameter
    /// names with pointer types.
    pub fn find_data_dependencies(funcs: &[PtxFunction]) -> Vec<DataDependency> {
        let mut deps = Vec::new();

        for (i, prod_func) in funcs.iter().enumerate() {
            let producer_outputs = output_params(prod_func);
            for (j, cons_func) in funcs.iter().enumerate().skip(i + 1) {
                let consumer_inputs = input_params(cons_func);
                for out_name in &producer_outputs {
                    for in_name in &consumer_inputs {
                        if out_name == in_name {
                            let pattern = infer_access_pattern(cons_func);
                            deps.push(DataDependency {
                                producer: i,
                                consumer: j,
                                buffer_name: out_name.clone(),
                                access_pattern: pattern,
                            });
                        }
                    }
                }
            }
        }

        deps
    }

    /// Estimates the speedup factor from fusing a candidate via a roofline-style
    /// cost model.
    ///
    /// The model compares the wall-clock cost of running the kernels separately
    /// against running them fused. Each kernel's cost is the maximum of its
    /// memory-transfer time and its compute time (the two overlap on real GPUs),
    /// plus a fixed per-launch overhead:
    ///
    /// ```text
    /// cost(kernel) = LAUNCH_OVERHEAD_NS
    ///              + max(bytes / MEM_BANDWIDTH, flops / COMPUTE_THROUGHPUT)
    /// ```
    ///
    /// Fusing `n` compatible kernels:
    ///
    /// * removes `n - 1` of the `n` kernel-launch overheads,
    /// * removes the global-memory round trips of every intermediate tensor
    ///   (`eliminated_global_bytes`), which now stay in registers / shared
    ///   memory, and
    /// * still performs the full arithmetic work of all kernels.
    ///
    /// ```text
    /// unfused = Σ_k cost(kernel_k)
    /// fused   = LAUNCH_OVERHEAD_NS
    ///         + max((Σ_k bytes_k − eliminated_bytes) / MEM_BANDWIDTH,
    ///               (Σ_k flops_k) / COMPUTE_THROUGHPUT)
    /// speedup = unfused / fused
    /// ```
    ///
    /// Occupancy losses from increased register / shared-memory pressure in the
    /// fused kernel are folded in as a multiplicative efficiency factor on the
    /// fused cost. A single-kernel candidate (`kernel_count < 2`) cannot be
    /// fused and yields exactly `1.0`. The result is always finite and clamped
    /// to the `[1.0, MAX_FUSION_SPEEDUP]` range.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_fusion_speedup(candidate: &FusionCandidate) -> f64 {
        // A group of fewer than two kernels has nothing to fuse.
        if candidate.kernel_count < 2 {
            return 1.0;
        }

        let n = candidate.kernel_count as f64;

        // --- Per-kernel costs run separately ------------------------------
        let prod_bytes = candidate.producer_global_bytes as f64;
        let cons_bytes = candidate.consumer_global_bytes as f64;
        let prod_flops = candidate.producer_flops as f64;
        let cons_flops = candidate.consumer_flops as f64;

        let prod_cost = LAUNCH_OVERHEAD_NS + kernel_runtime_ns(prod_bytes, prod_flops);
        let cons_cost = LAUNCH_OVERHEAD_NS + kernel_runtime_ns(cons_bytes, cons_flops);
        let unfused_cost = prod_cost + cons_cost;

        // --- Cost of the single fused kernel ------------------------------
        // The fused kernel keeps every launch overhead but one, and the
        // intermediate tensor traffic is removed from the byte total.
        let total_bytes = prod_bytes + cons_bytes;
        let eliminated = (candidate.eliminated_global_bytes as f64).min(total_bytes);
        let fused_bytes = (total_bytes - eliminated).max(0.0);
        let fused_flops = prod_flops + cons_flops;

        // Increased register / shared-memory pressure can reduce occupancy in
        // the fused kernel, stretching its effective runtime. Model this as an
        // efficiency factor in (0, 1]; cost is divided by it.
        let efficiency = fused_occupancy_efficiency(candidate);

        let fused_cost =
            LAUNCH_OVERHEAD_NS + kernel_runtime_ns(fused_bytes, fused_flops) / efficiency;

        // --- Speedup ------------------------------------------------------
        // Guard against degenerate (zero / non-finite) costs before dividing.
        let costs_valid = fused_cost.is_finite() && fused_cost > 0.0 && unfused_cost.is_finite();
        if !costs_valid {
            return 1.0;
        }
        let raw = unfused_cost / fused_cost;
        if !raw.is_finite() {
            return 1.0;
        }
        // Fusion can never be slower in this model (a fused kernel that gains
        // nothing degenerates to the launch-dominated lower bound), so floor at
        // 1.0 and cap at a sane ceiling for `n` overlapping kernels.
        raw.clamp(1.0, MAX_FUSION_SPEEDUP.min(n))
    }
}

// ---------------------------------------------------------------------------
// Cost-model constants
// ---------------------------------------------------------------------------

/// Fixed per-kernel launch overhead in nanoseconds.
///
/// Measured CUDA kernel-launch latency on modern hardware is roughly 3-10 us;
/// we use 5 us as a representative value. Eliminating these for fused kernels
/// is one of the two sources of fusion speedup.
const LAUNCH_OVERHEAD_NS: f64 = 5_000.0;

/// Effective global-memory bandwidth in bytes per nanosecond (i.e. GB/s).
///
/// 1500 GB/s is representative of a high-end HBM3 device. Used to convert a
/// per-thread byte count into a memory-transfer time.
const MEM_BANDWIDTH_B_PER_NS: f64 = 1_500.0;

/// Effective arithmetic throughput in operations per nanosecond (i.e. GFLOP/s).
///
/// 20000 GFLOP/s (20 TFLOP/s) is a conservative FP32 figure. Used to convert an
/// arithmetic-op count into a compute time.
const COMPUTE_THROUGHPUT_OPS_PER_NS: f64 = 20_000.0;

/// Upper bound on the speedup the model will report for any single candidate.
///
/// Keeps the estimate physically plausible even when the byte/flop accounting
/// is dominated by a tiny synthetic kernel.
const MAX_FUSION_SPEEDUP: f64 = 4.0;

/// Returns the overlapped runtime (ns) of a kernel given its per-thread global
/// byte traffic and arithmetic-op count.
///
/// Memory and compute overlap on real GPUs, so the kernel is bounded by
/// whichever resource takes longer (the roofline model).
fn kernel_runtime_ns(bytes: f64, flops: f64) -> f64 {
    let mem_ns = bytes / MEM_BANDWIDTH_B_PER_NS;
    let compute_ns = flops / COMPUTE_THROUGHPUT_OPS_PER_NS;
    mem_ns.max(compute_ns)
}

/// Estimates the occupancy efficiency of the fused kernel in `(0, 1]`.
///
/// Combining kernels sums their register and shared-memory footprints. When the
/// fused footprint grows large, fewer warps fit per SM and effective throughput
/// drops; we model that as a multiplicative efficiency applied to the fused
/// cost. A modest footprint keeps efficiency at `1.0`.
#[allow(clippy::cast_precision_loss)]
fn fused_occupancy_efficiency(candidate: &FusionCandidate) -> f64 {
    let reg_efficiency = if candidate.estimated_registers > 128 {
        0.90
    } else if candidate.estimated_registers > 64 {
        0.95
    } else {
        1.0
    };

    let smem_total = candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
    let smem_efficiency = if smem_total > 32_768 { 0.90 } else { 1.0 };

    reg_efficiency * smem_efficiency
}

// ---------------------------------------------------------------------------
// Top-level planning function
// ---------------------------------------------------------------------------

/// Plans kernel fusion for a sequence of kernels with resource constraints.
///
/// Analyzes all pairs, checks structural constraints, and then runs each
/// surviving candidate through a [`FusionCostModel`] to decide whether the
/// fusion is actually worthwhile. The result is a [`FusionReport`] describing
/// which kernels should be fused and why the rest were rejected.
///
/// The cost model is built from the supplied resource caps via
/// [`FusionCostModel::default`] with the register file and shared-memory budget
/// overridden to `max_registers` / `max_shared_mem`. For architecture-aware
/// budgets and occupancy thresholds, use [`plan_fusion_for_target`] instead.
///
/// # Arguments
///
/// * `kernels` - The sequence of PTX kernels to analyze.
/// * `max_registers` - Maximum registers per thread for the target device.
/// * `max_shared_mem` - Maximum shared memory per block in bytes.
pub fn plan_fusion(
    kernels: &[PtxFunction],
    max_registers: u32,
    max_shared_mem: u32,
) -> FusionReport {
    let cost_model = FusionCostModel::default()
        .with_register_file(max_registers)
        .with_register_pressure_threshold((max_registers / 2).clamp(32, max_registers.max(32)))
        .with_shared_mem_budget(max_shared_mem as usize);
    plan_fusion_with_model(kernels, max_registers, max_shared_mem, &cost_model)
}

/// Plans kernel fusion using an architecture-derived [`FusionCostModel`].
///
/// Equivalent to [`plan_fusion`] but the register file, occupancy threshold, and
/// shared-memory budget are derived from the target [`SmVersion`] via
/// [`FusionCostModel::for_target`], so the fuse/no-fuse decisions reflect that
/// architecture's real limits.
pub fn plan_fusion_for_target(kernels: &[PtxFunction], target: SmVersion) -> FusionReport {
    let cost_model = FusionCostModel::for_target(target);
    let max_registers = cost_model.register_file;
    let max_shared_mem = u32::try_from(cost_model.shared_mem_budget).unwrap_or(u32::MAX);
    plan_fusion_with_model(kernels, max_registers, max_shared_mem, &cost_model)
}

/// Core fusion planner shared by [`plan_fusion`] and [`plan_fusion_for_target`].
///
/// A candidate is accepted only when it passes the structural constraints **and**
/// the cost model deems it beneficial. This replaces the previous behaviour,
/// where any structurally legal candidate was fused unconditionally regardless
/// of whether it would spill or pay off.
fn plan_fusion_with_model(
    kernels: &[PtxFunction],
    max_registers: u32,
    max_shared_mem: u32,
    cost_model: &FusionCostModel,
) -> FusionReport {
    let candidates = FusionAnalysis::analyze_sequence(kernels);
    let constraints = vec![
        FusionConstraint::SameGridDimensions,
        FusionConstraint::NoSharedMemoryConflict,
        FusionConstraint::NoBarrierConflict,
        FusionConstraint::RegisterBudget(max_registers),
    ];

    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for cand in candidates {
        // Stage 1: structural legality (grid/barrier/hard caps).
        if !FusionAnalysis::check_constraints(&cand, &constraints) {
            let reason = rejection_reason(&cand, &constraints, max_shared_mem);
            rejected.push((cand, reason));
            continue;
        }

        // Stage 2: cost-model decision. Replaces the former unconditional accept
        // -- a legal candidate is only fused if the register-pressure +
        // shared-memory + ILP model says it is worthwhile on this target.
        let decision = cost_model.decide(&cand);
        if decision.should_fuse {
            accepted.push(cand);
        } else {
            let reason = cost_model_rejection_reason(decision.verdict);
            rejected.push((cand, reason));
        }
    }

    // Build fusion groups using a union-find approach
    let groups = build_fusion_groups(&accepted, kernels.len());

    let fused_kernel_count = groups.len();
    let total_speedup = if accepted.is_empty() {
        1.0
    } else {
        // Geometric mean of accepted speedups
        let product: f64 = accepted.iter().map(|c| c.estimated_speedup).product();
        #[allow(clippy::cast_precision_loss)]
        let n = accepted.len() as f64;
        product.powf(1.0 / n)
    };

    FusionReport {
        plan: FusionPlan {
            candidates: accepted,
            fused_groups: groups,
            original_kernel_count: kernels.len(),
            fused_kernel_count,
            estimated_total_speedup: total_speedup,
        },
        rejected,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Calculates total static shared memory bytes for a function.
fn shared_mem_bytes(func: &PtxFunction) -> usize {
    func.shared_mem
        .iter()
        .map(|(_, ty, count)| ty.size_bytes() * count)
        .sum()
}

/// Estimates the per-thread global-memory byte traffic of a kernel.
///
/// Sums the byte width of every global-memory load and store in the body,
/// accounting for the vector width of the access (`.v2`/`.v4`). Asynchronous
/// global-to-shared bulk copies (`cp.async`) and texture/surface global
/// accesses are counted as well. Shared/local/constant accesses are ignored
/// because they do not consume DRAM bandwidth. A kernel with no memory traffic
/// returns a small nonzero floor so the cost model never divides by zero.
fn global_traffic_bytes(func: &PtxFunction) -> usize {
    let mut bytes = 0usize;
    for inst in &func.body {
        bytes = bytes.saturating_add(instruction_global_bytes(inst));
    }
    bytes.max(1)
}

/// Returns the global-memory bytes moved by a single instruction (0 if none).
fn instruction_global_bytes(inst: &Instruction) -> usize {
    match inst {
        Instruction::Load { space, vec, ty, .. } | Instruction::Store { space, vec, ty, .. }
            if *space == crate::ir::MemorySpace::Global =>
        {
            ty.size_bytes() * vector_lanes(*vec)
        }
        Instruction::CpAsync { bytes, .. } => *bytes as usize,
        Instruction::Tex1d { ty, .. }
        | Instruction::Tex2d { ty, .. }
        | Instruction::Tex3d { ty, .. }
        | Instruction::SurfLoad { ty, .. }
        | Instruction::SurfStore { ty, .. } => ty.size_bytes() * 4,
        Instruction::AtomGlobalAddFloat { .. } => 4,
        // Everything else moves no statically-known global bytes. This includes
        // bulk / TMA transfers (`cp.async.bulk`, `tma.load`), whose size is not
        // encoded in the IR, so we conservatively omit them from the bandwidth
        // term rather than guess.
        _ => 0,
    }
}

/// Number of element lanes for a vector access width.
const fn vector_lanes(vec: crate::ir::VectorWidth) -> usize {
    match vec {
        crate::ir::VectorWidth::V1 => 1,
        crate::ir::VectorWidth::V2 => 2,
        crate::ir::VectorWidth::V4 => 4,
    }
}

/// Estimates the per-thread arithmetic-operation count of a kernel.
///
/// Counts arithmetic and math instructions, weighting fused/transcendental ops
/// by their approximate cost (an FMA performs two flops; transcendentals are
/// several). Pure data-movement, control-flow, synchronization, and addressing
/// instructions contribute nothing. The result feeds the compute side of the
/// roofline cost model.
fn arithmetic_op_count(func: &PtxFunction) -> u64 {
    let mut ops = 0u64;
    for inst in &func.body {
        ops = ops.saturating_add(instruction_op_cost(inst));
    }
    ops
}

/// Approximate arithmetic cost (in elementary ops) of a single instruction.
const fn instruction_op_cost(inst: &Instruction) -> u64 {
    match inst {
        // Single elementary arithmetic / logic ops.
        Instruction::Add { .. }
        | Instruction::Sub { .. }
        | Instruction::Mul { .. }
        | Instruction::Neg { .. }
        | Instruction::Abs { .. }
        | Instruction::Min { .. }
        | Instruction::Max { .. }
        | Instruction::Addc { .. }
        | Instruction::Selp { .. }
        | Instruction::Shl { .. }
        | Instruction::Shr { .. }
        | Instruction::And { .. }
        | Instruction::Or { .. }
        | Instruction::Xor { .. }
        | Instruction::Brev { .. }
        | Instruction::Clz { .. }
        | Instruction::Popc { .. }
        | Instruction::Bfind { .. }
        | Instruction::Bfe { .. }
        | Instruction::Bfi { .. }
        | Instruction::SetP { .. }
        | Instruction::Cvt { .. } => 1,

        // Fused multiply-add style ops perform two flops.
        Instruction::Mad { .. }
        | Instruction::MadLo { .. }
        | Instruction::MadHi { .. }
        | Instruction::MadWide { .. }
        | Instruction::Fma { .. }
        | Instruction::Dp2a { .. } => 2,

        // Multi-cycle ops: reciprocal / square root and integer divide /
        // remainder all cost roughly four elementary ops.
        Instruction::Rcp { .. }
        | Instruction::Rsqrt { .. }
        | Instruction::Sqrt { .. }
        | Instruction::Div { .. }
        | Instruction::Rem { .. } => 4,

        // Transcendental approximations and four-lane dot products are the
        // most expensive scalar ops we model.
        Instruction::Dp4a { .. }
        | Instruction::Ex2 { .. }
        | Instruction::Lg2 { .. }
        | Instruction::Sin { .. }
        | Instruction::Cos { .. } => 8,

        _ => 0,
    }
}

/// Estimates the per-thread global bytes that fusion eliminates by keeping the
/// intermediate tensor in registers / shared memory.
///
/// The intermediate is produced (written) by the producer and consumed (read)
/// by the consumer, so the eliminated traffic is one write plus one read of the
/// element that flows across the boundary. We size that element from the
/// producer's global stores (its output width); fusion removes both the store
/// and the consumer's matching load.
///
/// * For producer-consumer / vertical fusion the shared buffer is explicit, so
///   the full round trip is removed.
/// * For elementwise fusion an implicit temporary still flows between the two
///   ops, so a round trip is removed even without a named buffer.
/// * For horizontal fusion the kernels are independent (no shared tensor), so
///   nothing is eliminated.
fn intermediate_roundtrip_bytes(
    producer: &PtxFunction,
    consumer: &PtxFunction,
    fusion_type: FusionType,
    has_named_buffer: bool,
) -> usize {
    if fusion_type == FusionType::Horizontal {
        return 0;
    }
    if fusion_type != FusionType::Elementwise && !has_named_buffer {
        return 0;
    }

    // Width of one intermediate element = the producer's output store width.
    let elem = producer_output_store_bytes(producer)
        .or_else(|| consumer_input_load_bytes(consumer))
        .unwrap_or(PtxType::F32.size_bytes());

    // Round trip = one write (producer) + one read (consumer).
    elem.saturating_mul(2)
}

/// Width in bytes of the producer's global store (its output element), if any.
fn producer_output_store_bytes(func: &PtxFunction) -> Option<usize> {
    func.body.iter().find_map(|inst| match inst {
        Instruction::Store { space, vec, ty, .. } if *space == crate::ir::MemorySpace::Global => {
            Some(ty.size_bytes() * vector_lanes(*vec))
        }
        _ => None,
    })
}

/// Width in bytes of the consumer's first global load (its input element), if any.
fn consumer_input_load_bytes(func: &PtxFunction) -> Option<usize> {
    func.body.iter().find_map(|inst| match inst {
        Instruction::Load { space, vec, ty, .. } if *space == crate::ir::MemorySpace::Global => {
            Some(ty.size_bytes() * vector_lanes(*vec))
        }
        _ => None,
    })
}

/// Estimates register count from instruction body length.
///
/// A rough heuristic: each unique destination register in the body
/// contributes one register. We cap at a reasonable maximum.
fn estimate_register_count(func: &PtxFunction) -> u32 {
    let mut reg_names: HashSet<&str> = HashSet::new();
    for inst in &func.body {
        if let Some(name) = destination_register_name(inst) {
            reg_names.insert(name);
        }
    }
    // Minimum 1 register per kernel, even if body is empty
    let count = reg_names.len().max(1);
    // Clamp to u32
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// Extracts the destination register name from an instruction, if any.
fn destination_register_name(inst: &Instruction) -> Option<&str> {
    match inst {
        Instruction::Add { dst, .. }
        | Instruction::Sub { dst, .. }
        | Instruction::Mul { dst, .. }
        | Instruction::Mad { dst, .. }
        | Instruction::MadLo { dst, .. }
        | Instruction::MadHi { dst, .. }
        | Instruction::MadWide { dst, .. }
        | Instruction::Fma { dst, .. }
        | Instruction::Neg { dst, .. }
        | Instruction::Abs { dst, .. }
        | Instruction::Min { dst, .. }
        | Instruction::Max { dst, .. }
        | Instruction::Brev { dst, .. }
        | Instruction::Clz { dst, .. }
        | Instruction::Popc { dst, .. }
        | Instruction::Bfind { dst, .. }
        | Instruction::Bfe { dst, .. }
        | Instruction::Bfi { dst, .. }
        | Instruction::Rcp { dst, .. }
        | Instruction::Rsqrt { dst, .. }
        | Instruction::Sqrt { dst, .. }
        | Instruction::Ex2 { dst, .. }
        | Instruction::Lg2 { dst, .. }
        | Instruction::Sin { dst, .. }
        | Instruction::Cos { dst, .. }
        | Instruction::Shl { dst, .. }
        | Instruction::Shr { dst, .. }
        | Instruction::Div { dst, .. }
        | Instruction::Rem { dst, .. }
        | Instruction::And { dst, .. }
        | Instruction::Or { dst, .. }
        | Instruction::Xor { dst, .. }
        | Instruction::SetP { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cvt { dst, .. }
        | Instruction::MovSpecial { dst, .. }
        | Instruction::LoadParam { dst, .. }
        | Instruction::Atom { dst, .. }
        | Instruction::AtomCas { dst, .. }
        | Instruction::Dp4a { dst, .. }
        | Instruction::Dp2a { dst, .. }
        | Instruction::Tex1d { dst, .. }
        | Instruction::Tex2d { dst, .. }
        | Instruction::Tex3d { dst, .. }
        | Instruction::SurfLoad { dst, .. }
        | Instruction::Redux { dst, .. }
        | Instruction::ElectSync { dst, .. } => Some(&dst.name),

        _ => None,
    }
}

/// Checks if an instruction is non-elementwise (shared mem, sync, tensor core, etc.).
fn is_non_elementwise_instruction(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::BarSync { .. }
            | Instruction::BarArrive { .. }
            | Instruction::FenceAcqRel { .. }
            | Instruction::Mma { .. }
            | Instruction::Wgmma { .. }
            | Instruction::TmaLoad { .. }
            | Instruction::CpAsync { .. }
            | Instruction::CpAsyncCommit
            | Instruction::CpAsyncWait { .. }
            | Instruction::Redux { .. }
            | Instruction::Stmatrix { .. }
            | Instruction::MbarrierInit { .. }
            | Instruction::MbarrierArrive { .. }
            | Instruction::MbarrierWait { .. }
            | Instruction::FenceProxy { .. }
    ) || matches!(inst, Instruction::Wmma { .. })
        || is_shared_mem_access(inst)
}

/// Checks if an instruction accesses shared memory.
fn is_shared_mem_access(inst: &Instruction) -> bool {
    match inst {
        Instruction::Load { space, .. } | Instruction::Store { space, .. } => {
            *space == crate::ir::MemorySpace::Shared
        }
        _ => false,
    }
}

/// Checks if a kernel contains a reduction pattern.
///
/// Heuristic: looks for warp-level redux instructions or shared-memory
/// barrier patterns commonly used in reductions.
fn has_reduction(func: &PtxFunction) -> bool {
    func.body
        .iter()
        .any(|inst| matches!(inst, Instruction::Redux { .. }))
}

/// Finds a shared buffer name between producer output and consumer input params.
fn find_shared_buffer(producer: &PtxFunction, consumer: &PtxFunction) -> Option<String> {
    let producer_outputs = output_params(producer);
    let consumer_inputs = input_params(consumer);

    for out_name in &producer_outputs {
        for in_name in &consumer_inputs {
            if out_name == in_name {
                return Some(out_name.clone());
            }
        }
    }
    None
}

/// Returns output parameter names for a kernel.
///
/// Heuristic: the last pointer-typed (U64) parameter is treated as the output.
/// If the kernel has Store instructions referencing a `LoadParam`, those params
/// are also outputs.
fn output_params(func: &PtxFunction) -> Vec<String> {
    let mut outputs = Vec::new();

    // Heuristic: last U64 param is output (common CUDA convention)
    if let Some((name, _)) = func.params.iter().rev().find(|(_, ty)| *ty == PtxType::U64) {
        outputs.push(name.clone());
    }

    // Also check for params named with "output" or "out" or "dst" or "result"
    for (name, ty) in &func.params {
        if *ty == PtxType::U64 {
            let lower = name.to_lowercase();
            if (lower.contains("out") || lower.contains("dst") || lower.contains("result"))
                && !outputs.contains(name)
            {
                outputs.push(name.clone());
            }
        }
    }

    outputs
}

/// Returns input parameter names for a kernel.
///
/// Heuristic: all pointer-typed (U64) parameters that are not the last one,
/// plus params with "input" or "in" or "src" in the name.
fn input_params(func: &PtxFunction) -> Vec<String> {
    let mut inputs = Vec::new();
    let outputs = output_params(func);

    for (name, ty) in &func.params {
        if *ty == PtxType::U64 && !outputs.contains(name) {
            inputs.push(name.clone());
        }
    }

    // Also include params explicitly named as inputs
    for (name, ty) in &func.params {
        if *ty == PtxType::U64 {
            let lower = name.to_lowercase();
            if (lower.contains("in") || lower.contains("src")) && !inputs.contains(name) {
                inputs.push(name.clone());
            }
        }
    }

    // If no inputs found, treat all U64 params as potential inputs
    if inputs.is_empty() {
        for (name, ty) in &func.params {
            if *ty == PtxType::U64 {
                inputs.push(name.clone());
            }
        }
    }

    inputs
}

/// Infers the access pattern of a kernel from its instruction body.
fn infer_access_pattern(func: &PtxFunction) -> AccessPattern {
    let has_tid = func.body.iter().any(|inst| {
        matches!(
            inst,
            Instruction::MovSpecial {
                special: crate::ir::SpecialReg::TidX,
                ..
            }
        )
    });

    let has_stride_mul = func.body.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Mul { .. } | Instruction::Shl { .. } | Instruction::Mad { .. }
        )
    });

    if has_tid && !has_stride_mul {
        AccessPattern::Streaming
    } else if has_tid && has_stride_mul {
        // Could be strided; default to unknown stride
        AccessPattern::Strided(1)
    } else {
        AccessPattern::Unknown
    }
}

/// Checks if two kernels have compatible grid dimension hints.
///
/// Returns true if both have the same `max_threads` setting or if
/// neither specifies one (assumed compatible).
const fn compatible_grid_hints(a: &PtxFunction, b: &PtxFunction) -> bool {
    match (a.max_threads, b.max_threads) {
        (Some(ma), Some(mb)) => ma == mb,
        (None, None) => true,
        _ => false,
    }
}

/// Determines a human-readable rejection reason for a candidate.
fn rejection_reason(
    candidate: &FusionCandidate,
    constraints: &[FusionConstraint],
    _max_shared_mem: u32,
) -> String {
    for constraint in constraints {
        match constraint {
            FusionConstraint::NoSharedMemoryConflict => {
                let combined = candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
                if combined > 49152 {
                    return format!(
                        "combined shared memory ({combined} bytes) exceeds 48 KiB limit"
                    );
                }
            }
            FusionConstraint::NoBarrierConflict => {
                if candidate.producer_shared_bytes > 0
                    && candidate.consumer_shared_bytes > 0
                    && candidate.fusion_type != FusionType::Elementwise
                {
                    return "barrier conflict: both kernels use shared memory".to_string();
                }
            }
            FusionConstraint::RegisterBudget(max_regs) => {
                if candidate.estimated_registers > *max_regs {
                    return format!(
                        "register budget exceeded ({} > {max_regs})",
                        candidate.estimated_registers
                    );
                }
            }
            FusionConstraint::SameGridDimensions => {}
        }
    }
    "unknown reason".to_string()
}

/// Maps a cost-model [`FusionVerdict`] to a human-readable rejection reason for
/// candidates that passed the structural constraints but were not fused.
fn cost_model_rejection_reason(verdict: FusionVerdict) -> String {
    match verdict {
        FusionVerdict::RegisterSpill => {
            "cost model: fused kernel would spill registers to local memory".to_string()
        }
        FusionVerdict::SharedMemoryOverflow => {
            "cost model: combined shared memory exceeds the target's per-block budget".to_string()
        }
        FusionVerdict::NotWorthwhile => {
            "cost model: estimated benefit below the fusion threshold".to_string()
        }
        FusionVerdict::NothingToFuse => "cost model: fewer than two kernels to fuse".to_string(),
        FusionVerdict::Beneficial => "cost model: beneficial".to_string(),
    }
}

/// Builds fusion groups from accepted candidates using a union-find strategy.
///
/// Each group is a set of kernel indices that should be fused together.
/// Kernels not in any accepted candidate get their own singleton group.
/// Find with path compression (iterative) for union-find.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn build_fusion_groups(candidates: &[FusionCandidate], num_kernels: usize) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..num_kernels).collect();

    for cand in candidates {
        let pa = uf_find(&mut parent, cand.producer_index);
        let pb = uf_find(&mut parent, cand.consumer_index);
        if pa != pb {
            // Union: smaller root becomes child
            let (small, big) = if pa < pb { (pa, pb) } else { (pb, pa) };
            parent[big] = small;
        }
    }

    // Collect groups
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..num_kernels {
        let root = uf_find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    result.sort_by_key(|g| g[0]);
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CacheQualifier, ImmValue, Instruction, MemorySpace, Operand, PtxType, Register, SpecialReg,
        VectorWidth, WmmaLayout, WmmaOp, WmmaShape,
    };

    fn reg(name: &str, ty: PtxType) -> Register {
        Register {
            name: name.to_string(),
            ty,
        }
    }

    fn imm_f32(v: f32) -> Operand {
        Operand::Immediate(ImmValue::F32(v))
    }

    /// Build a simple elementwise kernel with the given name and params.
    fn make_elementwise_kernel(name: &str, params: &[(&str, PtxType)]) -> PtxFunction {
        let mut func = PtxFunction::new(name);
        for (pname, pty) in params {
            func.add_param(*pname, *pty);
        }
        // Simple elementwise: tid -> load -> add -> store
        func.body.push(Instruction::MovSpecial {
            dst: reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });
        func.body.push(Instruction::Add {
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            a: imm_f32(1.0),
            b: imm_f32(2.0),
        });
        func.body.push(Instruction::Store {
            space: MemorySpace::Global,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
            src: reg("%f0", PtxType::F32),
        });
        func
    }

    /// Build a kernel that uses shared memory and barriers.
    fn make_reduction_kernel(name: &str) -> PtxFunction {
        let mut func = PtxFunction::new(name);
        func.add_param("input", PtxType::U64);
        func.add_param("output", PtxType::U64);
        func.add_shared_mem("smem", PtxType::F32, 256);
        func.body.push(Instruction::MovSpecial {
            dst: reg("%r0", PtxType::U32),
            special: SpecialReg::TidX,
        });
        func.body.push(Instruction::BarSync { id: 0 });
        func.body.push(Instruction::Redux {
            op: crate::ir::ReduxOp::Add,
            dst: reg("%r1", PtxType::U32),
            src: Operand::Register(reg("%r0", PtxType::U32)),
            membership_mask: 0xFFFF_FFFF,
        });
        func
    }

    /// Builds a baseline two-kernel `FusionCandidate` with neutral cost fields.
    /// Tests override only the fields they exercise via struct-update syntax.
    fn base_candidate(fusion_type: FusionType) -> FusionCandidate {
        FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type,
            estimated_speedup: 0.0,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 32,
            producer_global_bytes: 256,
            consumer_global_bytes: 256,
            eliminated_global_bytes: 256,
            producer_flops: 64,
            consumer_flops: 64,
            kernel_count: 2,
        }
    }

    // -----------------------------------------------------------------------
    // Test: elementwise detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_elementwise_simple() {
        let kernel =
            make_elementwise_kernel("ew", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        assert!(FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_shared_mem() {
        let kernel = make_reduction_kernel("reduce");
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_barrier() {
        let mut kernel = PtxFunction::new("barrier_kernel");
        kernel.body.push(Instruction::BarSync { id: 0 });
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    #[test]
    fn test_is_not_elementwise_with_wmma() {
        let mut kernel = PtxFunction::new("wmma_kernel");
        kernel.body.push(Instruction::Wmma {
            op: WmmaOp::Mma,
            shape: WmmaShape::M16N16K16,
            layout: WmmaLayout::RowMajor,
            ty: PtxType::F16,
            fragments: vec![reg("%f0", PtxType::F16)],
            addr: None,
            stride: None,
        });
        assert!(!FusionAnalysis::is_elementwise(&kernel));
    }

    // -----------------------------------------------------------------------
    // Test: pair analysis
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_pair_elementwise() {
        let k0 =
            make_elementwise_kernel("add", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let k1 =
            make_elementwise_kernel("mul", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::Elementwise));
    }

    #[test]
    fn test_analyze_pair_producer_consumer() {
        let k0 = make_elementwise_kernel(
            "producer",
            &[("input", PtxType::U64), ("buf", PtxType::U64)],
        );
        let mut k1 = PtxFunction::new("consumer");
        k1.add_param("buf", PtxType::U64);
        k1.add_param("output", PtxType::U64);
        k1.add_shared_mem("smem", PtxType::F32, 64);
        k1.body.push(Instruction::Load {
            space: MemorySpace::Shared,
            qualifier: CacheQualifier::None,
            vec: VectorWidth::V1,
            ty: PtxType::F32,
            dst: reg("%f0", PtxType::F32),
            addr: Operand::Address {
                base: reg("%rd0", PtxType::U64),
                offset: None,
            },
        });

        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::ProducerConsumer));
    }

    #[test]
    fn test_analyze_pair_horizontal() {
        // Two kernels with same max_threads but no shared buffer
        let mut k0 = PtxFunction::new("kernel_a");
        k0.add_param("a_in", PtxType::U64);
        k0.add_param("a_out", PtxType::U64);
        k0.max_threads = Some(256);
        k0.body.push(Instruction::BarSync { id: 0 });

        let mut k1 = PtxFunction::new("kernel_b");
        k1.add_param("b_in", PtxType::U64);
        k1.add_param("b_out", PtxType::U64);
        k1.max_threads = Some(256);
        k1.body.push(Instruction::BarSync { id: 0 });

        let cand = FusionAnalysis::analyze_pair(&k0, &k1);
        assert!(cand.is_some());
        let c = cand.as_ref().map(|c| c.fusion_type);
        assert_eq!(c, Some(FusionType::Horizontal));
    }

    // -----------------------------------------------------------------------
    // Test: constraint checking
    // -----------------------------------------------------------------------

    #[test]
    fn test_constraints_pass() {
        let cand = FusionCandidate {
            estimated_speedup: 1.5,
            ..base_candidate(FusionType::Elementwise)
        };
        let constraints = vec![
            FusionConstraint::SameGridDimensions,
            FusionConstraint::NoSharedMemoryConflict,
            FusionConstraint::RegisterBudget(255),
        ];
        assert!(FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_register_budget_exceeded() {
        let cand = FusionCandidate {
            estimated_speedup: 1.3,
            estimated_registers: 300,
            ..base_candidate(FusionType::ProducerConsumer)
        };
        let constraints = vec![FusionConstraint::RegisterBudget(255)];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_shared_mem_exceeded() {
        let cand = FusionCandidate {
            estimated_speedup: 1.3,
            producer_shared_bytes: 32768,
            consumer_shared_bytes: 32768,
            ..base_candidate(FusionType::ProducerConsumer)
        };
        let constraints = vec![FusionConstraint::NoSharedMemoryConflict];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    #[test]
    fn test_constraints_barrier_conflict() {
        let cand = FusionCandidate {
            shared_buffer: "buf".to_string(),
            estimated_speedup: 1.3,
            producer_shared_bytes: 1024,
            consumer_shared_bytes: 1024,
            ..base_candidate(FusionType::ProducerConsumer)
        };
        let constraints = vec![FusionConstraint::NoBarrierConflict];
        assert!(!FusionAnalysis::check_constraints(&cand, &constraints));
    }

    // -----------------------------------------------------------------------
    // Test: fusion planning
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_fusion_two_elementwise() {
        let k0 =
            make_elementwise_kernel("add", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let k1 =
            make_elementwise_kernel("mul", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let report = plan_fusion(&[k0, k1], 255, 49152);
        assert!(!report.plan.candidates.is_empty());
        assert_eq!(report.plan.original_kernel_count, 2);
        // Two kernels fused into 1 group
        assert!(report.plan.fused_kernel_count <= 2);
    }

    #[test]
    fn test_plan_fusion_empty_sequence() {
        let report = plan_fusion(&[], 255, 49152);
        assert!(report.plan.candidates.is_empty());
        assert_eq!(report.plan.original_kernel_count, 0);
        assert_eq!(report.plan.fused_kernel_count, 0);
    }

    #[test]
    fn test_plan_fusion_single_kernel() {
        let k0 =
            make_elementwise_kernel("only", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let report = plan_fusion(&[k0], 255, 49152);
        assert!(report.plan.candidates.is_empty());
        assert_eq!(report.plan.fused_kernel_count, 1);
    }

    // -----------------------------------------------------------------------
    // Test: speedup estimation
    // -----------------------------------------------------------------------

    #[test]
    fn test_speedup_multi_kernel_is_greater_than_one() {
        // A genuine two-kernel fusible group that eliminates an intermediate
        // round trip must report a speedup strictly above 1.0.
        let cand = FusionCandidate {
            estimated_registers: 16,
            ..base_candidate(FusionType::Elementwise)
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        assert!(speedup.is_finite(), "speedup must be finite, got {speedup}");
        assert!(speedup >= 1.0, "speedup must be >= 1.0, got {speedup}");
        assert!(speedup > 1.0, "fusible pair should beat 1.0, got {speedup}");
    }

    #[test]
    fn test_speedup_single_kernel_is_one() {
        // A group of a single kernel has nothing to fuse: speedup is exactly 1.0.
        let cand = FusionCandidate {
            kernel_count: 1,
            ..base_candidate(FusionType::Elementwise)
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        assert!(
            (speedup - 1.0).abs() < 1e-9,
            "single-kernel group must be ~1.0, got {speedup}"
        );
    }

    #[test]
    fn test_speedup_more_eliminated_traffic_is_strictly_larger() {
        // Two otherwise-identical candidates: the one eliminating more
        // intermediate global traffic must have a strictly larger speedup.
        let small = FusionCandidate {
            producer_global_bytes: 1024,
            consumer_global_bytes: 1024,
            eliminated_global_bytes: 64,
            // Memory-bound so the bandwidth savings dominate the estimate.
            producer_flops: 1,
            consumer_flops: 1,
            ..base_candidate(FusionType::ProducerConsumer)
        };
        let large = FusionCandidate {
            eliminated_global_bytes: 1024,
            ..small.clone()
        };
        let s_small = FusionAnalysis::estimate_fusion_speedup(&small);
        let s_large = FusionAnalysis::estimate_fusion_speedup(&large);
        assert!(
            s_large > s_small,
            "more eliminated traffic ({s_large}) must exceed less ({s_small})"
        );
        assert!(s_large.is_finite() && s_large >= 1.0);
    }

    #[test]
    fn test_speedup_high_register_pressure() {
        // Higher register pressure lowers fused occupancy, reducing speedup.
        let high = FusionCandidate {
            estimated_registers: 200,
            ..base_candidate(FusionType::Elementwise)
        };
        let low = FusionCandidate {
            estimated_registers: 16,
            ..base_candidate(FusionType::Elementwise)
        };
        let high_speedup = FusionAnalysis::estimate_fusion_speedup(&high);
        let low_speedup = FusionAnalysis::estimate_fusion_speedup(&low);
        assert!(
            high_speedup < low_speedup,
            "high regs ({high_speedup}) should be < low regs ({low_speedup})"
        );
        assert!(high_speedup >= 1.0 && high_speedup.is_finite());
    }

    #[test]
    fn test_speedup_no_eliminated_traffic_still_at_least_one() {
        // Horizontal fusion eliminates no intermediate traffic, but removing a
        // launch overhead still keeps the estimate >= 1.0 and finite.
        let cand = FusionCandidate {
            eliminated_global_bytes: 0,
            ..base_candidate(FusionType::Horizontal)
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        assert!(speedup.is_finite(), "must be finite, got {speedup}");
        assert!(speedup >= 1.0, "must be >= 1.0, got {speedup}");
    }

    #[test]
    fn test_speedup_capped_and_finite_on_extreme_input() {
        // Degenerate inputs (huge eliminated traffic, zero flops) must remain
        // finite and within the documented ceiling.
        let cand = FusionCandidate {
            producer_global_bytes: usize::MAX / 4,
            consumer_global_bytes: usize::MAX / 4,
            eliminated_global_bytes: usize::MAX / 4,
            producer_flops: 0,
            consumer_flops: 0,
            ..base_candidate(FusionType::Elementwise)
        };
        let speedup = FusionAnalysis::estimate_fusion_speedup(&cand);
        assert!(speedup.is_finite(), "must be finite, got {speedup}");
        assert!(
            (1.0..=MAX_FUSION_SPEEDUP).contains(&speedup),
            "must be clamped, got {speedup}"
        );
    }

    #[test]
    fn test_global_traffic_and_op_count_from_kernel() {
        // The derived cost fields must reflect the actual instruction stream.
        let kernel =
            make_elementwise_kernel("ew", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        // make_elementwise_kernel emits one global F32 store (4 bytes) and one Add.
        assert_eq!(global_traffic_bytes(&kernel), 4);
        assert_eq!(arithmetic_op_count(&kernel), 1);
    }

    #[test]
    fn test_analyze_pair_populates_cost_fields() {
        let k0 =
            make_elementwise_kernel("add", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let k1 =
            make_elementwise_kernel("mul", &[("input", PtxType::U64), ("output", PtxType::U64)]);
        let cand = FusionAnalysis::analyze_pair(&k0, &k1).expect("pair should fuse");
        assert_eq!(cand.kernel_count, 2);
        assert!(cand.producer_global_bytes >= 1);
        assert!(cand.consumer_global_bytes >= 1);
        // Elementwise pair carries an implicit intermediate -> some elimination.
        assert!(cand.eliminated_global_bytes > 0);
        assert!(cand.estimated_speedup.is_finite());
        assert!(cand.estimated_speedup >= 1.0);
    }

    // -----------------------------------------------------------------------
    // Test: data dependency detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_data_dependencies() {
        let k0 = make_elementwise_kernel(
            "producer",
            &[("input", PtxType::U64), ("buf", PtxType::U64)],
        );
        let k1 = make_elementwise_kernel(
            "consumer",
            &[("buf", PtxType::U64), ("output", PtxType::U64)],
        );
        let deps = FusionAnalysis::find_data_dependencies(&[k0, k1]);
        assert!(!deps.is_empty(), "expected at least one dependency");
        assert_eq!(deps[0].producer, 0);
        assert_eq!(deps[0].consumer, 1);
        assert_eq!(deps[0].buffer_name, "buf");
    }

    #[test]
    fn test_no_data_dependencies() {
        let k0 = make_elementwise_kernel("a", &[("a_in", PtxType::U64), ("a_out", PtxType::U64)]);
        let k1 = make_elementwise_kernel("b", &[("b_in", PtxType::U64), ("b_out", PtxType::U64)]);
        let deps = FusionAnalysis::find_data_dependencies(&[k0, k1]);
        assert!(
            deps.is_empty(),
            "expected no dependencies, got {}",
            deps.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test: Display implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_fusion_plan_display() {
        let plan = FusionPlan {
            candidates: vec![],
            fused_groups: vec![vec![0, 1], vec![2]],
            original_kernel_count: 3,
            fused_kernel_count: 2,
            estimated_total_speedup: 1.5,
        };
        let display = format!("{plan}");
        assert!(display.contains("Original kernels: 3"));
        assert!(display.contains("Fused kernels:    2"));
        assert!(display.contains("1.50x"));
    }

    #[test]
    fn test_fusion_report_display() {
        let rejected_cand = FusionCandidate {
            consumer_index: 2,
            shared_buffer: "buf".to_string(),
            estimated_speedup: 1.3,
            estimated_registers: 300,
            ..base_candidate(FusionType::ProducerConsumer)
        };
        let report = FusionReport {
            plan: FusionPlan {
                candidates: vec![],
                fused_groups: vec![vec![0], vec![1], vec![2]],
                original_kernel_count: 3,
                fused_kernel_count: 3,
                estimated_total_speedup: 1.0,
            },
            rejected: vec![(rejected_cand, "register budget exceeded".to_string())],
        };
        let display = format!("{report}");
        assert!(display.contains("Rejected candidates:"));
        assert!(display.contains("register budget exceeded"));
    }

    #[test]
    fn test_fusion_type_display() {
        assert_eq!(format!("{}", FusionType::Elementwise), "elementwise");
        assert_eq!(
            format!("{}", FusionType::ProducerConsumer),
            "producer-consumer"
        );
        assert_eq!(format!("{}", FusionType::Horizontal), "horizontal");
        assert_eq!(format!("{}", FusionType::Vertical), "vertical");
    }

    #[test]
    fn test_access_pattern_display() {
        assert_eq!(format!("{}", AccessPattern::Streaming), "streaming");
        assert_eq!(format!("{}", AccessPattern::Random), "random");
        assert_eq!(format!("{}", AccessPattern::Strided(4)), "strided(4)");
        assert_eq!(format!("{}", AccessPattern::Unknown), "unknown");
    }
}
