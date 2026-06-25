//! Standalone `cp.async` (asynchronous global-to-shared copy) PTX generator.
//!
//! This module provides [`CpAsyncGenerator`], a focused code generator that
//! emits the PTX instruction sequences used to stage data from global memory
//! into shared memory through the Ampere+ asynchronous copy engine. It is the
//! building block that multi-stage software-pipelined kernels (GEMM, attention,
//! convolution) rely on to overlap memory transfers with computation.
//!
//! # Cache policies
//!
//! The PTX ISA exposes two cache hint variants for `cp.async`:
//!
//! * **`cp.async.cg.global`** -- *cache-global*. Bypasses the L1 cache and
//!   caches only in L2. The 16-byte form additionally takes an `L2::128B`
//!   cache-eviction-priority hint (`cp.async.cg.global.L2::128B`). This is the
//!   right policy for large streaming staging buffers that will not be re-read
//!   from L1.
//! * **`cp.async.ca.global`** -- *cache-all*. Caches the copied data in both L1
//!   and L2. Preferred for smaller transfers (4 / 8 byte) and data with temporal
//!   locality. The `.cg` variant is only legal for a 16-byte copy size, so a
//!   request for `cg` at 4 or 8 bytes is transparently downgraded to `ca`.
//!
//! # Multi-stage pipeline
//!
//! A software pipeline issues several `cp.async` copies, groups them with
//! `cp.async.commit_group`, and then overlaps computation on already-arrived
//! stages while waiting on the oldest outstanding group via
//! `cp.async.wait_group N` (where `N` is the number of groups still allowed to
//! be in flight). [`CpAsyncGenerator::generate_pipeline_loop`] emits a complete
//! prologue + steady-state loop + epilogue for a configurable stage count.
//!
//! # Architecture fallback
//!
//! `cp.async` requires `sm_80` (Ampere) or newer. On `sm_75` (Turing) the
//! generator emits a *synchronous* `ld.global` -> `st.shared` fallback that has
//! the same memory effect (without the async overlap), so the same template can
//! target the full architecture range.
//!
//! # Example
//!
//! ```
//! use oxicuda_ptx::templates::cp_async_gen::{CpAsyncGenerator, CpAsyncCachePolicy};
//! use oxicuda_ptx::arch::SmVersion;
//!
//! // 16-byte cache-global staging copies, 3-stage pipeline.
//! let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
//!     .expect("valid configuration");
//!
//! let copy = generator.generate_copy(SmVersion::Sm80, "%shared_ptr", "%global_ptr");
//! assert!(copy.contains("cp.async.cg.global.L2::128B"));
//!
//! let loop_body = generator.generate_pipeline_loop(SmVersion::Sm80);
//! assert!(loop_body.contains("cp.async.commit_group"));
//! assert!(loop_body.contains("cp.async.wait_group"));
//! ```

use std::fmt::Write as FmtWrite;

use crate::arch::SmVersion;
use crate::error::PtxGenError;

/// The minimum SM version that supports the `cp.async` instruction family.
pub const CP_ASYNC_MIN_SM: SmVersion = SmVersion::Sm80;

/// The only copy size (in bytes) for which the `.cg` cache-global hint and its
/// `L2::128B` eviction-priority modifier are legal in PTX.
const CG_REQUIRED_BYTES: u32 = 16;

/// Cache policy hint applied to an emitted `cp.async` copy.
///
/// Maps directly onto the PTX `.ca` / `.cg` instruction modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpAsyncCachePolicy {
    /// `cp.async.ca.global` -- cache in both L1 and L2 (cache-all). Legal for
    /// 4, 8, and 16 byte copies.
    CacheAll,
    /// `cp.async.cg.global` -- bypass L1, cache only in L2 (cache-global). The
    /// 16-byte form carries the `L2::128B` eviction-priority hint. Only legal
    /// for a 16-byte copy size.
    CacheGlobal,
}

impl CpAsyncCachePolicy {
    /// Returns the PTX modifier token for this policy (`"ca"` or `"cg"`).
    #[must_use]
    pub const fn modifier(self) -> &'static str {
        match self {
            Self::CacheAll => "ca",
            Self::CacheGlobal => "cg",
        }
    }

    /// Returns the policy actually usable for the given copy size.
    ///
    /// The `.cg` hint is only legal for 16-byte copies, so a `CacheGlobal`
    /// request at a smaller size is downgraded to `CacheAll`.
    #[must_use]
    pub const fn resolve_for_bytes(self, bytes: u32) -> Self {
        match self {
            // The `.cg` cache-global hint is only legal at the 16-byte copy
            // size; every other case (smaller cache-global, or any cache-all)
            // resolves to the always-legal `.ca` form.
            Self::CacheGlobal if bytes == CG_REQUIRED_BYTES => Self::CacheGlobal,
            Self::CacheGlobal | Self::CacheAll => Self::CacheAll,
        }
    }
}

/// Generator for `cp.async` global-to-shared copy sequences and multi-stage
/// software pipelines.
///
/// Construct with [`CpAsyncGenerator::new`], which validates the copy size and
/// stage count, then call [`CpAsyncGenerator::generate_copy`] for a single copy
/// or [`CpAsyncGenerator::generate_pipeline_loop`] for a full pipelined loop.
#[derive(Debug, Clone)]
pub struct CpAsyncGenerator {
    /// Requested cache policy. The effective policy per copy is resolved from
    /// this and the copy size via [`CpAsyncCachePolicy::resolve_for_bytes`].
    pub cache_policy: CpAsyncCachePolicy,
    /// Bytes copied per `cp.async` instruction. Must be one of 4, 8, or 16.
    pub bypass_bytes: u32,
    /// Number of pipeline stages (depth of the software pipeline). Must be >= 2.
    pub stages: u32,
}

impl CpAsyncGenerator {
    /// Creates a new generator, validating its configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::GenerationFailed`] if:
    /// - `bypass_bytes` is not one of 4, 8, or 16 (the only sizes `cp.async`
    ///   accepts).
    /// - `stages` is less than 2 (a pipeline needs at least a producer stage
    ///   and a consumer stage to overlap anything).
    pub fn new(
        cache_policy: CpAsyncCachePolicy,
        bypass_bytes: u32,
        stages: u32,
    ) -> Result<Self, PtxGenError> {
        if !matches!(bypass_bytes, 4 | 8 | 16) {
            return Err(PtxGenError::GenerationFailed(format!(
                "cp.async copy size must be 4, 8, or 16 bytes, got {bypass_bytes}"
            )));
        }
        if stages < 2 {
            return Err(PtxGenError::GenerationFailed(format!(
                "cp.async pipeline needs at least 2 stages, got {stages}"
            )));
        }
        Ok(Self {
            cache_policy,
            bypass_bytes,
            stages,
        })
    }

    /// Returns the cache policy effective for this generator's copy size.
    ///
    /// This applies the `.cg`-requires-16-bytes rule, so it may differ from the
    /// originally requested [`Self::cache_policy`].
    #[must_use]
    pub const fn effective_policy(&self) -> CpAsyncCachePolicy {
        self.cache_policy.resolve_for_bytes(self.bypass_bytes)
    }

    /// Returns the number of `cp.async` groups that the steady-state loop keeps
    /// in flight, i.e. the argument to `cp.async.wait_group`.
    ///
    /// For an `n`-stage pipeline the loop commits one group per iteration and
    /// waits until only `n - 1` groups remain outstanding before consuming the
    /// oldest stage, overlapping `n - 1` in-flight transfers with compute.
    #[must_use]
    pub const fn in_flight_groups(&self) -> u32 {
        self.stages - 1
    }

    /// Returns `true` if the target architecture supports `cp.async` natively.
    #[must_use]
    pub fn supports_async(sm: SmVersion) -> bool {
        sm >= CP_ASYNC_MIN_SM && sm.capabilities().has_cp_async
    }

    /// Builds the cache-policy-qualified opcode for a single copy, including the
    /// `L2::128B` eviction hint when the cache-global 16-byte form is selected.
    ///
    /// Examples: `cp.async.cg.global.L2::128B`, `cp.async.ca.global`.
    #[must_use]
    pub fn copy_opcode(&self) -> String {
        let policy = self.effective_policy();
        match policy {
            CpAsyncCachePolicy::CacheGlobal => {
                // Only reachable at 16 bytes (resolve_for_bytes guarantees it),
                // where the L2::128B eviction-priority hint is legal.
                "cp.async.cg.global.L2::128B".to_string()
            }
            CpAsyncCachePolicy::CacheAll => "cp.async.ca.global".to_string(),
        }
    }

    /// Generates a single global-to-shared copy.
    ///
    /// `dst_shared` and `src_global` are PTX register operands (e.g.
    /// `"%shared_ptr"`, `"%global_ptr"`) already holding the destination shared
    /// address and source global address respectively.
    ///
    /// On `sm_80`+ this emits a single `cp.async` instruction. On pre-Ampere
    /// targets it emits a synchronous `ld.global` -> `st.shared` fallback with
    /// the same memory effect (see [`Self::generate_sync_fallback`]).
    #[must_use]
    pub fn generate_copy(&self, sm: SmVersion, dst_shared: &str, src_global: &str) -> String {
        if Self::supports_async(sm) {
            format!(
                "{} [{dst_shared}], [{src_global}], {};\n",
                self.copy_opcode(),
                self.bypass_bytes
            )
        } else {
            self.generate_sync_fallback(dst_shared, src_global)
        }
    }

    /// Emits a synchronous load/store fallback for pre-`sm_80` targets.
    ///
    /// `cp.async` does not exist before Ampere, so the data is staged with a
    /// blocking `ld.global` into a scratch register followed by `st.shared`.
    /// The register width matches the copy size (`.b32` / `.b64` / `.v4.b32`).
    #[must_use]
    pub fn generate_sync_fallback(&self, dst_shared: &str, src_global: &str) -> String {
        let mut out = String::with_capacity(128);
        out.push_str("    // async copy unavailable (< sm_80): synchronous fallback\n");
        match self.bypass_bytes {
            4 => {
                let _ = writeln!(out, "    ld.global.b32 %cpa_tmp_b32, [{src_global}];");
                let _ = writeln!(out, "    st.shared.b32 [{dst_shared}], %cpa_tmp_b32;");
            }
            8 => {
                let _ = writeln!(out, "    ld.global.b64 %cpa_tmp_b64, [{src_global}];");
                let _ = writeln!(out, "    st.shared.b64 [{dst_shared}], %cpa_tmp_b64;");
            }
            // 16 bytes: a vector-of-4 32-bit load/store pair.
            _ => {
                let _ = writeln!(
                    out,
                    "    ld.global.v4.b32 {{%cpa_v0, %cpa_v1, %cpa_v2, %cpa_v3}}, [{src_global}];"
                );
                let _ = writeln!(
                    out,
                    "    st.shared.v4.b32 [{dst_shared}], {{%cpa_v0, %cpa_v1, %cpa_v2, %cpa_v3}};"
                );
            }
        }
        out
    }

    /// Emits a `cp.async.commit_group;` barrier that closes the current group.
    #[must_use]
    pub fn generate_commit(sm: SmVersion) -> String {
        if Self::supports_async(sm) {
            "    cp.async.commit_group;\n".to_string()
        } else {
            // Synchronous fallback has no in-flight groups to commit.
            "    // cp.async.commit_group elided (< sm_80)\n".to_string()
        }
    }

    /// Emits a `cp.async.wait_group N;` that drains all but `n` groups.
    #[must_use]
    pub fn generate_wait(sm: SmVersion, n: u32) -> String {
        if Self::supports_async(sm) {
            format!("    cp.async.wait_group {n};\n")
        } else {
            "    // cp.async.wait_group elided (< sm_80)\n".to_string()
        }
    }

    /// Emits a full `cp.async.wait_group 0;` (wait for all outstanding copies).
    #[must_use]
    pub fn generate_wait_all(sm: SmVersion) -> String {
        Self::generate_wait(sm, 0)
    }

    /// Generates a complete multi-stage software-pipelined copy loop.
    ///
    /// The emitted PTX has three sections:
    ///
    /// 1. **Prologue** -- issues `stages - 1` copies (one per warm-up stage),
    ///    each followed by its own `cp.async.commit_group`, so the pipeline is
    ///    primed before the steady state begins.
    /// 2. **Steady-state loop** -- for each main iteration: waits until only
    ///    `stages - 1` groups remain in flight (`cp.async.wait_group`), issues
    ///    the next copy into the rotating stage buffer, commits it, and marks
    ///    where the consumer would process the just-arrived stage.
    /// 3. **Epilogue** -- a final `cp.async.wait_group 0;` drains every
    ///    remaining in-flight group before the kernel proceeds.
    ///
    /// The loop uses symbolic register names (`%stage_dst`, `%stage_src`) as
    /// placeholders for the per-iteration rotating addresses; a caller embeds
    /// this fragment after computing those addresses for its own access
    /// pattern. On pre-`sm_80` targets the body degrades to the synchronous
    /// fallback with the group barriers elided.
    #[must_use]
    pub fn generate_pipeline_loop(&self, sm: SmVersion) -> String {
        let mut out = String::with_capacity(1024);
        let async_ok = Self::supports_async(sm);
        let wait_n = self.in_flight_groups();

        let _ = writeln!(
            out,
            "    // === cp.async {}-stage software pipeline ({}, {} B/copy) ===",
            self.stages,
            self.effective_policy().modifier(),
            self.bypass_bytes
        );

        // --- Prologue: prime stages-1 warm-up stages --------------------------
        let _ = writeln!(out, "    // Prologue: prime {wait_n} pipeline stage(s)");
        for stage in 0..wait_n {
            let _ = writeln!(out, "    // prologue stage {stage}");
            out.push_str("    ");
            out.push_str(&self.generate_copy(sm, "%stage_dst", "%stage_src"));
            out.push_str(&Self::generate_commit(sm));
        }

        // --- Steady-state loop ------------------------------------------------
        let _ = writeln!(out, "$CP_ASYNC_LOOP:");
        let _ = writeln!(
            out,
            "    // Wait until only {wait_n} group(s) remain in flight, then consume the oldest"
        );
        out.push_str(&Self::generate_wait(sm, wait_n));
        if async_ok {
            let _ = writeln!(
                out,
                "    bar.sync 0; // ensure the awaited stage is visible to all threads"
            );
        }
        let _ = writeln!(out, "    // <consume arrived stage here>");
        let _ = writeln!(out, "    // Issue the next stage into the rotating buffer");
        out.push_str("    ");
        out.push_str(&self.generate_copy(sm, "%stage_dst", "%stage_src"));
        out.push_str(&Self::generate_commit(sm));
        let _ = writeln!(out, "    // loop-control: @%p_more bra $CP_ASYNC_LOOP;");

        // --- Epilogue: drain everything --------------------------------------
        let _ = writeln!(out, "    // Epilogue: drain all remaining in-flight groups");
        out.push_str(&Self::generate_wait_all(sm));
        if async_ok {
            let _ = writeln!(out, "    bar.sync 0;");
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction / validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_valid_sizes() {
        for bytes in [4u32, 8, 16] {
            let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, bytes, 2);
            assert!(generator.is_ok(), "{bytes} bytes should be valid");
        }
    }

    #[test]
    fn test_new_rejects_bad_size() {
        for bytes in [0u32, 1, 2, 3, 5, 12, 32] {
            let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, bytes, 2);
            assert!(generator.is_err(), "{bytes} bytes must be rejected");
        }
    }

    #[test]
    fn test_new_rejects_too_few_stages() {
        for stages in [0u32, 1] {
            let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, stages);
            assert!(generator.is_err(), "{stages} stages must be rejected");
        }
        assert!(CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 2).is_ok());
    }

    // -----------------------------------------------------------------------
    // Cache-policy resolution and opcode formation (semantic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_modifier_tokens() {
        assert_eq!(CpAsyncCachePolicy::CacheAll.modifier(), "ca");
        assert_eq!(CpAsyncCachePolicy::CacheGlobal.modifier(), "cg");
    }

    #[test]
    fn test_cg_resolves_to_ca_below_16_bytes() {
        // cache-global is only legal at 16 bytes; smaller sizes downgrade.
        assert_eq!(
            CpAsyncCachePolicy::CacheGlobal.resolve_for_bytes(4),
            CpAsyncCachePolicy::CacheAll
        );
        assert_eq!(
            CpAsyncCachePolicy::CacheGlobal.resolve_for_bytes(8),
            CpAsyncCachePolicy::CacheAll
        );
        assert_eq!(
            CpAsyncCachePolicy::CacheGlobal.resolve_for_bytes(16),
            CpAsyncCachePolicy::CacheGlobal
        );
    }

    #[test]
    fn test_effective_policy_follows_size() {
        let cg16 = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        assert_eq!(cg16.effective_policy(), CpAsyncCachePolicy::CacheGlobal);

        let cg8 = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 8, 3)
            .expect("valid cp.async config");
        assert_eq!(cg8.effective_policy(), CpAsyncCachePolicy::CacheAll);
    }

    #[test]
    fn test_copy_opcode_cg16_has_l2_hint() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        assert_eq!(generator.copy_opcode(), "cp.async.cg.global.L2::128B");
    }

    #[test]
    fn test_copy_opcode_ca_has_no_l2_hint() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, 16, 3)
            .expect("valid cp.async config");
        assert_eq!(generator.copy_opcode(), "cp.async.ca.global");
        assert!(!generator.copy_opcode().contains("L2::128B"));
    }

    #[test]
    fn test_copy_opcode_cg8_downgrades_to_ca() {
        // A cache-global request at 8 bytes is illegal; opcode must be `ca`.
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 8, 3)
            .expect("valid cp.async config");
        assert_eq!(generator.copy_opcode(), "cp.async.ca.global");
        assert!(!generator.copy_opcode().contains("cg"));
    }

    // -----------------------------------------------------------------------
    // Single-copy emission (structural + semantic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_copy_cg16_structure() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        let ptx = generator.generate_copy(SmVersion::Sm80, "%sh", "%gl");
        // Correct opcode for cache-policy + size.
        assert!(ptx.contains("cp.async.cg.global.L2::128B"));
        // Operand order: [dst_shared], [src_global], bytes
        assert!(ptx.contains("[%sh], [%gl], 16;"));
    }

    #[test]
    fn test_generate_copy_ca_sizes() {
        for bytes in [4u32, 8, 16] {
            let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, bytes, 3)
                .expect("valid cp.async config");
            let ptx = generator.generate_copy(SmVersion::Sm90, "%sh", "%gl");
            assert!(ptx.contains("cp.async.ca.global"));
            assert!(
                ptx.contains(&format!("[%sh], [%gl], {bytes};")),
                "byte size {bytes} must appear in operand list: {ptx}"
            );
        }
    }

    #[test]
    fn test_generate_copy_supported_on_all_async_archs() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 2)
            .expect("valid cp.async config");
        for sm in [
            SmVersion::Sm80,
            SmVersion::Sm86,
            SmVersion::Sm89,
            SmVersion::Sm90,
            SmVersion::Sm90a,
            SmVersion::Sm100,
            SmVersion::Sm120,
        ] {
            let ptx = generator.generate_copy(sm, "%sh", "%gl");
            assert!(
                ptx.contains("cp.async"),
                "sm {sm} should emit a cp.async copy"
            );
            assert!(!ptx.contains("fallback"), "sm {sm} should not fall back");
        }
    }

    // -----------------------------------------------------------------------
    // Pre-sm_80 fallback (semantic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_supports_async_gate() {
        assert!(!CpAsyncGenerator::supports_async(SmVersion::Sm75));
        assert!(CpAsyncGenerator::supports_async(SmVersion::Sm80));
        assert!(CpAsyncGenerator::supports_async(SmVersion::Sm90));
    }

    #[test]
    fn test_fallback_on_sm75_no_cp_async() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        let ptx = generator.generate_copy(SmVersion::Sm75, "%sh", "%gl");
        // No async instruction must be emitted on Turing.
        assert!(!ptx.contains("cp.async"));
        // Must use a synchronous load + shared store with the right width.
        assert!(ptx.contains("ld.global.v4.b32"));
        assert!(ptx.contains("st.shared.v4.b32"));
    }

    #[test]
    fn test_fallback_widths_match_size() {
        let cases = [
            (4u32, "ld.global.b32", "st.shared.b32"),
            (8u32, "ld.global.b64", "st.shared.b64"),
            (16u32, "ld.global.v4.b32", "st.shared.v4.b32"),
        ];
        for (bytes, ld, st) in cases {
            let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, bytes, 2)
                .expect("valid cp.async config");
            let ptx = generator.generate_copy(SmVersion::Sm75, "%sh", "%gl");
            assert!(
                ptx.contains(ld),
                "{bytes} B fallback should use {ld}: {ptx}"
            );
            assert!(
                ptx.contains(st),
                "{bytes} B fallback should use {st}: {ptx}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Commit / wait barriers (semantic)
    // -----------------------------------------------------------------------

    #[test]
    fn test_commit_and_wait_async() {
        assert_eq!(
            CpAsyncGenerator::generate_commit(SmVersion::Sm80),
            "    cp.async.commit_group;\n"
        );
        assert_eq!(
            CpAsyncGenerator::generate_wait(SmVersion::Sm80, 2),
            "    cp.async.wait_group 2;\n"
        );
        assert_eq!(
            CpAsyncGenerator::generate_wait_all(SmVersion::Sm90),
            "    cp.async.wait_group 0;\n"
        );
    }

    #[test]
    fn test_commit_and_wait_elided_pre_ampere() {
        assert!(
            !CpAsyncGenerator::generate_commit(SmVersion::Sm75).contains("cp.async.commit_group;")
        );
        assert!(
            !CpAsyncGenerator::generate_wait(SmVersion::Sm75, 2).contains("cp.async.wait_group 2;")
        );
    }

    #[test]
    fn test_in_flight_groups_is_stages_minus_one() {
        assert_eq!(
            CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, 16, 2)
                .expect("valid cp.async config")
                .in_flight_groups(),
            1
        );
        assert_eq!(
            CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, 16, 4)
                .expect("valid cp.async config")
                .in_flight_groups(),
            3
        );
    }

    // -----------------------------------------------------------------------
    // Pipeline loop (structural + semantic staging count)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_loop_has_all_phases() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        let ptx = generator.generate_pipeline_loop(SmVersion::Sm80);
        // Prologue / steady-state / epilogue all present.
        assert!(ptx.contains("Prologue"));
        assert!(ptx.contains("$CP_ASYNC_LOOP:"));
        assert!(ptx.contains("Epilogue"));
        // Pipeline staging barriers present.
        assert!(ptx.contains("cp.async.commit_group"));
        assert!(ptx.contains("cp.async.wait_group"));
        // The opcode used in the loop reflects the cache policy + size.
        assert!(ptx.contains("cp.async.cg.global.L2::128B"));
    }

    #[test]
    fn test_pipeline_loop_wait_group_uses_stages_minus_one() {
        // A 4-stage pipeline must steady-state-wait on `stages - 1 == 3` groups.
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, 16, 4)
            .expect("valid cp.async config");
        let ptx = generator.generate_pipeline_loop(SmVersion::Sm80);
        assert!(
            ptx.contains("cp.async.wait_group 3;"),
            "steady-state wait must drain to stages-1 groups: {ptx}"
        );
        // Epilogue drains everything.
        assert!(ptx.contains("cp.async.wait_group 0;"));
    }

    #[test]
    fn test_pipeline_prologue_commit_count_matches_warmup_stages() {
        // The prologue must issue exactly `stages - 1` commit_groups, plus one
        // more in the steady-state body = `stages` total in the fragment.
        let stages = 3u32;
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, stages)
            .expect("valid cp.async config");
        let ptx = generator.generate_pipeline_loop(SmVersion::Sm80);
        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert_eq!(
            commit_count, stages as usize,
            "expected {stages} commit_group instructions, got {commit_count}"
        );
    }

    #[test]
    fn test_pipeline_loop_falls_back_pre_ampere() {
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheGlobal, 16, 3)
            .expect("valid cp.async config");
        let ptx = generator.generate_pipeline_loop(SmVersion::Sm75);
        // No async machinery on Turing.
        assert!(!ptx.contains("cp.async.cg"));
        assert!(!ptx.contains("cp.async.commit_group;"));
        assert!(!ptx.contains("cp.async.wait_group 0;"));
        // But the synchronous staging copy is still emitted.
        assert!(ptx.contains("ld.global.v4.b32"));
        assert!(ptx.contains("st.shared.v4.b32"));
    }

    #[test]
    fn test_pipeline_loop_two_stage_minimal() {
        // The minimal 2-stage pipeline: one warm-up copy, wait_group 1.
        let generator = CpAsyncGenerator::new(CpAsyncCachePolicy::CacheAll, 8, 2)
            .expect("valid cp.async config");
        let ptx = generator.generate_pipeline_loop(SmVersion::Sm86);
        assert!(ptx.contains("cp.async.wait_group 1;"));
        let commit_count = ptx.matches("cp.async.commit_group;").count();
        assert_eq!(commit_count, 2, "2-stage fragment commits prologue + body");
    }
}
