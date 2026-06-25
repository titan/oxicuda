//! Cost model that decides whether fusing two adjacent pointwise kernels pays
//! off on a specific target architecture.
//!
//! [`crate::analysis::kernel_fusion`] identifies *candidate* fusions and estimates a
//! roofline speedup, but historically the planner accepted every structurally
//! legal candidate unconditionally -- it fused as long as the combined register
//! and shared-memory totals stayed under a flat hard cap, with no notion of
//! whether the fused kernel would actually run faster or whether it would spill
//! on the chosen SM. [`FusionCostModel`] replaces that unconditional accept with
//! a real decision.
//!
//! # The heuristic
//!
//! Fusing two pointwise kernels concatenates their bodies into one launch. That
//! is profitable when it removes an intermediate global round trip and a kernel
//! launch *without* pushing the fused kernel past the architecture's register or
//! shared-memory budget. The model weighs three forces:
//!
//! 1. **Register pressure.** The fused kernel's live-register estimate must stay
//!    within the per-thread register file *and* below the threshold that would
//!    cut occupancy below the target. Past the hardware limit the kernel spills
//!    to local memory, which is far slower than the round trip fusion removed --
//!    so a candidate that would spill is always refused.
//! 2. **Shared memory.** The combined static shared-memory footprint must fit in
//!    the per-block shared-memory budget for the target SM; exceeding it makes
//!    the fused launch illegal.
//! 3. **Instruction-level parallelism (ILP).** Concatenating two independent
//!    pointwise bodies lengthens the dependency-free instruction window, letting
//!    the scheduler hide more latency. A higher combined arithmetic-op count
//!    relative to register pressure indicates healthy ILP and reinforces the
//!    fuse decision; a body that is almost all long dependency chains with high
//!    register pressure does not.
//!
//! The three signals are combined into a single benefit score; fusion is chosen
//! when the candidate is *feasible* (fits registers and shared memory) and the
//! score clears a configurable threshold.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::analysis::fusion_cost_model::FusionCostModel;
//! use oxicuda_ptx::analysis::kernel_fusion::{FusionCandidate, FusionType};
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let model = FusionCostModel::for_target(SmVersion::Sm80);
//!
//! // A light pointwise pair fits comfortably and should fuse.
//! let light = FusionCandidate {
//!     producer_index: 0,
//!     consumer_index: 1,
//!     shared_buffer: String::new(),
//!     fusion_type: FusionType::Elementwise,
//!     estimated_speedup: 1.4,
//!     producer_shared_bytes: 0,
//!     consumer_shared_bytes: 0,
//!     estimated_registers: 40,
//!     producer_global_bytes: 256,
//!     consumer_global_bytes: 256,
//!     eliminated_global_bytes: 256,
//!     producer_flops: 64,
//!     consumer_flops: 64,
//!     kernel_count: 2,
//! };
//! assert!(model.decide(&light).should_fuse);
//!
//! // A pair whose combined registers exceed the SM file must be refused.
//! let heavy = FusionCandidate { estimated_registers: 400, ..light.clone() };
//! assert!(!model.decide(&heavy).should_fuse);
//! ```

use std::fmt;

use crate::analysis::kernel_fusion::FusionCandidate;
use crate::arch::SmVersion;

// ---------------------------------------------------------------------------
// Decision result
// ---------------------------------------------------------------------------

/// Why a fusion candidate was accepted or rejected by [`FusionCostModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionVerdict {
    /// Fusion is beneficial: it fits the budgets and clears the benefit
    /// threshold.
    Beneficial,
    /// The combined kernel would exceed the per-thread register file (it would
    /// spill to local memory), so fusion is refused.
    RegisterSpill,
    /// The combined static shared-memory footprint exceeds the per-block budget
    /// for the target architecture, so the fused launch is illegal.
    SharedMemoryOverflow,
    /// The candidate fits the budgets but the modelled benefit does not clear
    /// the threshold (e.g. occupancy loss outweighs the saved round trip).
    NotWorthwhile,
    /// The candidate covers fewer than two kernels: nothing to fuse.
    NothingToFuse,
}

impl FusionVerdict {
    /// Returns `true` only for [`FusionVerdict::Beneficial`].
    #[must_use]
    pub const fn is_fuse(self) -> bool {
        matches!(self, Self::Beneficial)
    }
}

impl fmt::Display for FusionVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Beneficial => write!(f, "beneficial"),
            Self::RegisterSpill => write!(f, "register spill"),
            Self::SharedMemoryOverflow => write!(f, "shared-memory overflow"),
            Self::NotWorthwhile => write!(f, "not worthwhile"),
            Self::NothingToFuse => write!(f, "nothing to fuse"),
        }
    }
}

/// The outcome of evaluating a single fusion candidate against the cost model.
#[derive(Debug, Clone)]
pub struct FusionDecision {
    /// Whether the planner should fuse this candidate.
    pub should_fuse: bool,
    /// The classification explaining the decision.
    pub verdict: FusionVerdict,
    /// Combined live-register estimate of the fused kernel.
    pub combined_registers: u32,
    /// Combined static shared-memory footprint of the fused kernel, in bytes.
    pub combined_shared_bytes: usize,
    /// The modelled instruction-level-parallelism score (higher is healthier).
    pub ilp_score: f64,
    /// The aggregate benefit score the candidate produced; compared against the
    /// model's `benefit_threshold` to decide worthwhileness.
    pub benefit_score: f64,
}

impl fmt::Display for FusionDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "fuse={} ({}): regs={}, smem={}B, ilp={:.2}, benefit={:.2}",
            self.should_fuse,
            self.verdict,
            self.combined_registers,
            self.combined_shared_bytes,
            self.ilp_score,
            self.benefit_score
        )
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// A register-pressure + shared-memory + ILP cost model that decides whether to
/// fuse two adjacent pointwise kernels for a particular target SM.
///
/// Construct with [`FusionCostModel::for_target`] to derive the budgets from an
/// [`SmVersion`], or build one explicitly via the field-setting methods for
/// custom tuning. Then call [`FusionCostModel::decide`] per candidate.
#[derive(Debug, Clone, Copy)]
pub struct FusionCostModel {
    /// Hard per-thread register-file limit. A fused kernel whose live-register
    /// estimate exceeds this spills to local memory and is always refused.
    pub register_file: u32,
    /// Register count above which occupancy starts dropping. Kept below
    /// `register_file`; crossing it taxes the benefit score but does not by
    /// itself refuse the fusion.
    pub register_pressure_threshold: u32,
    /// Per-block shared-memory budget in bytes. A combined footprint above this
    /// makes the fused launch illegal.
    pub shared_mem_budget: usize,
    /// Minimum benefit score a feasible candidate must reach to be fused.
    pub benefit_threshold: f64,
}

impl FusionCostModel {
    /// Builds a cost model with budgets derived from the target architecture.
    ///
    /// The register file is the canonical 256 architectural registers per thread
    /// (the PTX `.reg` ceiling and the hardware maximum on all supported SMs).
    /// The occupancy-pressure threshold is set so that at least four warps can
    /// co-reside (a fused kernel exceeding it loses occupancy). The shared-memory
    /// budget is taken from [`SmVersion::max_shared_mem_per_block`].
    #[must_use]
    pub fn for_target(sm: SmVersion) -> Self {
        // 256 architectural registers per thread is the PTX/hardware ceiling;
        // beyond it the compiler must spill.
        let register_file = MAX_REGISTERS_PER_THREAD;

        // Occupancy threshold: registers-per-thread that still allow a healthy
        // number of resident warps. Derived from the SM's register file size
        // (64 K 32-bit registers) divided by the warps we want resident.
        let regs_per_sm = REGISTERS_PER_SM;
        let warps_target = OCCUPANCY_WARPS_TARGET * sm.warp_size();
        let pressure_threshold = (regs_per_sm / warps_target.max(1))
            .min(register_file - 1)
            .max(32);

        Self {
            register_file,
            register_pressure_threshold: pressure_threshold,
            shared_mem_budget: sm.max_shared_mem_per_block() as usize,
            benefit_threshold: DEFAULT_BENEFIT_THRESHOLD,
        }
    }

    /// Overrides the hard register-file limit (builder style).
    #[must_use]
    pub const fn with_register_file(mut self, regs: u32) -> Self {
        self.register_file = regs;
        self
    }

    /// Overrides the occupancy register-pressure threshold (builder style).
    #[must_use]
    pub const fn with_register_pressure_threshold(mut self, regs: u32) -> Self {
        self.register_pressure_threshold = regs;
        self
    }

    /// Overrides the shared-memory budget in bytes (builder style).
    #[must_use]
    pub const fn with_shared_mem_budget(mut self, bytes: usize) -> Self {
        self.shared_mem_budget = bytes;
        self
    }

    /// Overrides the minimum benefit threshold (builder style).
    #[must_use]
    pub const fn with_benefit_threshold(mut self, threshold: f64) -> Self {
        self.benefit_threshold = threshold;
        self
    }

    /// Computes the instruction-level-parallelism score of a fused candidate.
    ///
    /// Concatenating two independent pointwise bodies widens the dependency-free
    /// instruction window, so more arithmetic ops *per live register* means the
    /// scheduler can hide more latency. The score is the combined arithmetic-op
    /// count divided by the combined register pressure, normalised so that the
    /// neutral point (roughly one op per register) is `1.0`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ilp_score(candidate: &FusionCandidate) -> f64 {
        let ops = (candidate.producer_flops + candidate.consumer_flops) as f64;
        let regs = f64::from(candidate.estimated_registers.max(1));
        // Ops-per-register; >1 indicates a wide, latency-tolerant window.
        ops / regs
    }

    /// Evaluates a fusion candidate and returns the decision with diagnostics.
    ///
    /// The candidate is fused only when it is *feasible* on the target (its
    /// combined registers fit the file and its combined shared memory fits the
    /// budget) **and** the modelled benefit clears [`Self::benefit_threshold`].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn decide(&self, candidate: &FusionCandidate) -> FusionDecision {
        let combined_registers = candidate.estimated_registers;
        let combined_shared_bytes =
            candidate.producer_shared_bytes + candidate.consumer_shared_bytes;
        let ilp_score = Self::ilp_score(candidate);

        // A single-kernel group has nothing to fuse.
        if candidate.kernel_count < 2 {
            return FusionDecision {
                should_fuse: false,
                verdict: FusionVerdict::NothingToFuse,
                combined_registers,
                combined_shared_bytes,
                ilp_score,
                benefit_score: 0.0,
            };
        }

        // --- Hard feasibility gates ---------------------------------------
        // Register spill: exceeding the architectural register file forces the
        // compiler to spill to local memory, which costs far more than the round
        // trip fusion removes. Always refuse.
        if combined_registers > self.register_file {
            return FusionDecision {
                should_fuse: false,
                verdict: FusionVerdict::RegisterSpill,
                combined_registers,
                combined_shared_bytes,
                ilp_score,
                benefit_score: 0.0,
            };
        }

        // Shared-memory overflow makes the fused launch illegal on this SM.
        if combined_shared_bytes > self.shared_mem_budget {
            return FusionDecision {
                should_fuse: false,
                verdict: FusionVerdict::SharedMemoryOverflow,
                combined_registers,
                combined_shared_bytes,
                ilp_score,
                benefit_score: 0.0,
            };
        }

        // --- Benefit scoring ----------------------------------------------
        let benefit_score = self.benefit_score(candidate, ilp_score);

        if benefit_score >= self.benefit_threshold {
            FusionDecision {
                should_fuse: true,
                verdict: FusionVerdict::Beneficial,
                combined_registers,
                combined_shared_bytes,
                ilp_score,
                benefit_score,
            }
        } else {
            FusionDecision {
                should_fuse: false,
                verdict: FusionVerdict::NotWorthwhile,
                combined_registers,
                combined_shared_bytes,
                ilp_score,
                benefit_score,
            }
        }
    }

    /// Convenience wrapper returning just the fuse/no-fuse boolean.
    #[must_use]
    pub fn should_fuse(&self, candidate: &FusionCandidate) -> bool {
        self.decide(candidate).should_fuse
    }

    /// Computes the aggregate benefit score that drives the worthwhileness
    /// decision for a feasible candidate.
    ///
    /// The score starts from the roofline speedup the analysis already estimated
    /// (`estimated_speedup`), then applies multiplicative penalties:
    ///
    /// * an **occupancy penalty** when combined register pressure crosses
    ///   [`Self::register_pressure_threshold`] (fewer resident warps), and
    /// * a **shared-memory penalty** as the combined footprint approaches the
    ///   budget (fewer co-resident blocks),
    ///
    /// and a small **ILP bonus** when the dependency-free window is wide enough
    /// to tolerate the extra latency. The result is the net expected gain factor
    /// from fusing; values at or below `1.0` mean fusion is not worth it.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn benefit_score(&self, candidate: &FusionCandidate, ilp_score: f64) -> f64 {
        // Base gain from the roofline model (>= 1.0 by construction).
        let base = candidate.estimated_speedup.max(1.0);

        // Occupancy penalty from register pressure: linear falloff from 1.0 at
        // the threshold down to OCCUPANCY_FLOOR at the register file ceiling.
        let reg_penalty = if candidate.estimated_registers <= self.register_pressure_threshold {
            1.0
        } else {
            let over = f64::from(candidate.estimated_registers - self.register_pressure_threshold);
            let span = f64::from(
                self.register_file
                    .saturating_sub(self.register_pressure_threshold)
                    .max(1),
            );
            (over / span)
                .mul_add(-(1.0 - OCCUPANCY_FLOOR), 1.0)
                .max(OCCUPANCY_FLOOR)
        };

        // Shared-memory penalty: full strength while under half the budget, then
        // linear falloff toward OCCUPANCY_FLOOR at the budget.
        let smem = (candidate.producer_shared_bytes + candidate.consumer_shared_bytes) as f64;
        let budget = self.shared_mem_budget.max(1) as f64;
        let smem_penalty = if smem <= budget * 0.5 {
            1.0
        } else {
            let over = smem - budget * 0.5;
            let span = budget * 0.5;
            (over / span)
                .mul_add(-(1.0 - OCCUPANCY_FLOOR), 1.0)
                .max(OCCUPANCY_FLOOR)
        };

        // ILP bonus: a wide dependency-free window (ops-per-register above the
        // neutral point) lets the scheduler hide latency, modestly improving the
        // realised gain. Clamped so a pathological op count cannot dominate.
        let ilp_bonus =
            1.0 + ((ilp_score - ILP_NEUTRAL) * ILP_BONUS_WEIGHT).clamp(0.0, ILP_BONUS_CAP);

        base * reg_penalty * smem_penalty * ilp_bonus
    }
}

impl Default for FusionCostModel {
    /// A neutral default tuned for Ampere (`sm_80`).
    fn default() -> Self {
        Self::for_target(SmVersion::Sm80)
    }
}

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// The PTX architectural maximum of registers a single thread may use before
/// the compiler is forced to spill to local memory.
const MAX_REGISTERS_PER_THREAD: u32 = 256;

/// Number of 32-bit registers in an SM's register file (64 Ki on every
/// supported architecture). Used to derive the occupancy pressure threshold.
const REGISTERS_PER_SM: u32 = 65_536;

/// Number of warps we would like to keep co-resident on an SM when deriving the
/// register-pressure threshold from the register file size.
const OCCUPANCY_WARPS_TARGET: u32 = 8;

/// Lower bound applied to the occupancy penalties so a single feasible candidate
/// near the budget never collapses the benefit score to zero.
const OCCUPANCY_FLOOR: f64 = 0.5;

/// The benefit score a feasible candidate must reach to be fused. Just above
/// 1.0 so that any genuine net gain fuses while neutral/negative cases do not.
const DEFAULT_BENEFIT_THRESHOLD: f64 = 1.05;

/// The ops-per-register value treated as ILP-neutral (one op per live register).
const ILP_NEUTRAL: f64 = 1.0;

/// Weight converting ops-per-register above neutral into a benefit bonus.
const ILP_BONUS_WEIGHT: f64 = 0.05;

/// Maximum additive ILP bonus, so a huge op count cannot dominate the score.
const ILP_BONUS_CAP: f64 = 0.25;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::kernel_fusion::FusionType;

    /// A baseline two-kernel candidate that comfortably fits and benefits.
    fn light_candidate() -> FusionCandidate {
        FusionCandidate {
            producer_index: 0,
            consumer_index: 1,
            shared_buffer: String::new(),
            fusion_type: FusionType::Elementwise,
            estimated_speedup: 1.4,
            producer_shared_bytes: 0,
            consumer_shared_bytes: 0,
            estimated_registers: 40,
            producer_global_bytes: 256,
            consumer_global_bytes: 256,
            eliminated_global_bytes: 256,
            producer_flops: 64,
            consumer_flops: 64,
            kernel_count: 2,
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_for_target_budgets() {
        let m80 = FusionCostModel::for_target(SmVersion::Sm80);
        assert_eq!(m80.register_file, 256);
        assert_eq!(m80.shared_mem_budget, 163_840);
        assert!(m80.register_pressure_threshold < m80.register_file);
        assert!(m80.register_pressure_threshold >= 32);

        let m75 = FusionCostModel::for_target(SmVersion::Sm75);
        // Turing has a smaller shared-memory budget than Ampere.
        assert_eq!(m75.shared_mem_budget, 65_536);
        assert!(m75.shared_mem_budget < m80.shared_mem_budget);
    }

    #[test]
    fn test_builder_overrides() {
        let m = FusionCostModel::for_target(SmVersion::Sm80)
            .with_register_file(200)
            .with_register_pressure_threshold(96)
            .with_shared_mem_budget(8192)
            .with_benefit_threshold(2.0);
        assert_eq!(m.register_file, 200);
        assert_eq!(m.register_pressure_threshold, 96);
        assert_eq!(m.shared_mem_budget, 8192);
        assert!((m.benefit_threshold - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_default_is_ampere() {
        let d = FusionCostModel::default();
        let a = FusionCostModel::for_target(SmVersion::Sm80);
        assert_eq!(d.register_file, a.register_file);
        assert_eq!(d.shared_mem_budget, a.shared_mem_budget);
    }

    // -----------------------------------------------------------------------
    // Fuse when beneficial
    // -----------------------------------------------------------------------

    #[test]
    fn test_fuses_low_pressure_pair() {
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let decision = model.decide(&light_candidate());
        assert!(decision.should_fuse, "{decision}");
        assert_eq!(decision.verdict, FusionVerdict::Beneficial);
        assert!(decision.benefit_score >= model.benefit_threshold);
    }

    #[test]
    fn test_should_fuse_wrapper_agrees_with_decide() {
        let model = FusionCostModel::for_target(SmVersion::Sm90);
        let cand = light_candidate();
        assert_eq!(model.should_fuse(&cand), model.decide(&cand).should_fuse);
    }

    #[test]
    fn test_fuses_with_modest_shared_memory() {
        // Some shared memory, but well under the SM budget: still beneficial.
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let cand = FusionCandidate {
            producer_shared_bytes: 4096,
            consumer_shared_bytes: 4096,
            ..light_candidate()
        };
        assert!(model.decide(&cand).should_fuse);
    }

    // -----------------------------------------------------------------------
    // Refuse on register spill
    // -----------------------------------------------------------------------

    #[test]
    fn test_refuses_register_spill() {
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let cand = FusionCandidate {
            estimated_registers: 400, // > 256 register file
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert!(!decision.should_fuse);
        assert_eq!(decision.verdict, FusionVerdict::RegisterSpill);
    }

    #[test]
    fn test_register_spill_exactly_at_limit_is_allowed() {
        // At the file ceiling (not over) the kernel does not spill.
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let cand = FusionCandidate {
            estimated_registers: model.register_file,
            // Make the benefit obviously positive so feasibility is the only
            // question under test.
            estimated_speedup: 2.0,
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert_ne!(decision.verdict, FusionVerdict::RegisterSpill);
    }

    // -----------------------------------------------------------------------
    // Refuse on shared-memory overflow
    // -----------------------------------------------------------------------

    #[test]
    fn test_refuses_shared_memory_overflow() {
        // Tight budget; combined footprint exceeds it.
        let model = FusionCostModel::for_target(SmVersion::Sm75).with_shared_mem_budget(32_768);
        let cand = FusionCandidate {
            producer_shared_bytes: 20_000,
            consumer_shared_bytes: 20_000, // 40 KiB > 32 KiB budget
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert!(!decision.should_fuse);
        assert_eq!(decision.verdict, FusionVerdict::SharedMemoryOverflow);
    }

    #[test]
    fn test_shared_memory_just_fits() {
        let model = FusionCostModel::for_target(SmVersion::Sm80).with_shared_mem_budget(16_384);
        let cand = FusionCandidate {
            producer_shared_bytes: 8_000,
            consumer_shared_bytes: 8_000, // 16 KiB == budget, legal
            estimated_speedup: 2.0,
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert_ne!(decision.verdict, FusionVerdict::SharedMemoryOverflow);
    }

    // -----------------------------------------------------------------------
    // Refuse when feasible but not worthwhile
    // -----------------------------------------------------------------------

    #[test]
    fn test_refuses_when_not_worthwhile() {
        // Fits the budgets, but no modelled gain (speedup ~1.0) -> below
        // threshold -> NotWorthwhile, not a spill/overflow.
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let cand = FusionCandidate {
            estimated_speedup: 1.0,
            producer_flops: 0,
            consumer_flops: 0,
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert!(!decision.should_fuse);
        assert_eq!(decision.verdict, FusionVerdict::NotWorthwhile);
    }

    #[test]
    fn test_high_register_pressure_taxes_benefit() {
        // Same speedup, but pressure above the occupancy threshold lowers the
        // realised benefit relative to a low-pressure pair.
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let low = FusionCandidate {
            estimated_registers: 32,
            ..light_candidate()
        };
        let high = FusionCandidate {
            estimated_registers: model.register_file - 1,
            ..light_candidate()
        };
        let low_b = model.decide(&low).benefit_score;
        let high_b = model.decide(&high).benefit_score;
        assert!(
            high_b < low_b,
            "high pressure benefit ({high_b}) must be < low ({low_b})"
        );
    }

    // -----------------------------------------------------------------------
    // Nothing to fuse
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_kernel_nothing_to_fuse() {
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let cand = FusionCandidate {
            kernel_count: 1,
            ..light_candidate()
        };
        let decision = model.decide(&cand);
        assert!(!decision.should_fuse);
        assert_eq!(decision.verdict, FusionVerdict::NothingToFuse);
    }

    // -----------------------------------------------------------------------
    // ILP scoring
    // -----------------------------------------------------------------------

    #[test]
    fn test_ilp_score_scales_with_ops_per_register() {
        let few_ops = FusionCandidate {
            producer_flops: 8,
            consumer_flops: 8,
            estimated_registers: 64,
            ..light_candidate()
        };
        let many_ops = FusionCandidate {
            producer_flops: 256,
            consumer_flops: 256,
            estimated_registers: 64,
            ..light_candidate()
        };
        assert!(
            FusionCostModel::ilp_score(&many_ops) > FusionCostModel::ilp_score(&few_ops),
            "more ops per register must raise the ILP score"
        );
    }

    #[test]
    fn test_ilp_bonus_lifts_benefit() {
        // Two feasible candidates with identical speedup but very different ILP:
        // the high-ILP one should score at least as high.
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let low_ilp = FusionCandidate {
            producer_flops: 1,
            consumer_flops: 1,
            estimated_registers: 64,
            ..light_candidate()
        };
        let high_ilp = FusionCandidate {
            producer_flops: 1024,
            consumer_flops: 1024,
            estimated_registers: 64,
            ..light_candidate()
        };
        let low_b = model.decide(&low_ilp).benefit_score;
        let high_b = model.decide(&high_ilp).benefit_score;
        assert!(high_b >= low_b, "ILP bonus must not reduce the benefit");
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", FusionVerdict::Beneficial), "beneficial");
        assert_eq!(
            format!("{}", FusionVerdict::RegisterSpill),
            "register spill"
        );
        assert_eq!(
            format!("{}", FusionVerdict::SharedMemoryOverflow),
            "shared-memory overflow"
        );
        assert_eq!(
            format!("{}", FusionVerdict::NotWorthwhile),
            "not worthwhile"
        );
        assert_eq!(
            format!("{}", FusionVerdict::NothingToFuse),
            "nothing to fuse"
        );
    }

    #[test]
    fn test_decision_display_contains_fields() {
        let model = FusionCostModel::for_target(SmVersion::Sm80);
        let s = format!("{}", model.decide(&light_candidate()));
        assert!(s.contains("fuse=true"));
        assert!(s.contains("regs="));
        assert!(s.contains("smem="));
        assert!(s.contains("ilp="));
    }

    #[test]
    fn test_verdict_is_fuse_helper() {
        assert!(FusionVerdict::Beneficial.is_fuse());
        assert!(!FusionVerdict::RegisterSpill.is_fuse());
        assert!(!FusionVerdict::NotWorthwhile.is_fuse());
    }
}
