//! PTX-parseable SM occupancy from register declarations.
//!
//! The CUDA driver's `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`
//! needs the *registers-per-thread* figure to give an accurate occupancy
//! answer.  On a live GPU that number comes from `ptxas`; offline, the only
//! source of truth is the PTX text itself — every kernel begins with a set of
//! `.reg` directives that declare its virtual register file.
//!
//! This module parses those `.reg` directives **per `.entry`** and feeds the
//! resulting register count into the CPU-side [`OccupancyCalculator`] from
//! [`crate::occupancy_ext`].  It is entirely host-side — no GPU is required —
//! which makes it useful for build-time kernel auto-tuning and for occupancy
//! reports in CI.
//!
//! # PTX register declaration grammar
//!
//! Inside a kernel body, virtual registers are declared with lines such as:
//!
//! ```ptx
//! .reg .f32   %f<10>;   // 10 single-precision registers
//! .reg .b32   %r<5>;    //  5 untyped 32-bit registers
//! .reg .pred  %p<3>;    //  3 predicate registers
//! .reg .f64   %fd<4>;   //  4 double-precision registers (each 2x 32-bit)
//! ```
//!
//! The bracketed `<N>` is the *vectorised* count: it declares `%f0 .. %f9`.
//! A bare declaration without brackets (e.g. `.reg .u32 %tid;`) declares a
//! single register.
//!
//! The hardware register file is 32-bit-word addressed, so wide types
//! (`.f64`, `.b64`, `.s64`, `.u64`) occupy **two** 32-bit slots each, while
//! `.pred` registers do **not** consume general-purpose registers at all (they
//! live in a separate condition-code file).  [`PtxRegisterUsage`] models all of
//! this so the occupancy estimate matches what `ptxas` would compute for the
//! `-O0` (no register coalescing) lower bound.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_driver::occupancy_register_count::OccupancyFromPtx;
//!
//! let ptx = r#"
//! .version 8.0
//! .target sm_90
//! .visible .entry saxpy() {
//!     .reg .f32   %f<8>;
//!     .reg .b32   %r<4>;
//!     .reg .pred  %p<2>;
//!     ret;
//! }
//! "#;
//!
//! let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid PTX");
//! // saxpy uses 8 f32 + 4 b32 = 12 general-purpose 32-bit registers.
//! assert_eq!(from_ptx.kernel("saxpy").unwrap().registers_per_thread(), 12);
//!
//! // Feed straight into the CPU occupancy model for an sm_90 device.
//! let est = from_ptx.estimate_for("saxpy", 9, 0, 256, 0).unwrap();
//! assert!(est.occupancy_ratio > 0.0);
//! ```

use crate::error::{CudaError, CudaResult};
use crate::occupancy_ext::{DeviceOccupancyInfo, OccupancyCalculator, OccupancyEstimate};

// ---------------------------------------------------------------------------
// PtxRegisterWidth
// ---------------------------------------------------------------------------

/// Number of 32-bit hardware register slots a PTX type occupies per element.
///
/// The NVIDIA hardware register file is addressed in 32-bit words.  A 64-bit
/// value therefore consumes two slots; sub-word values (`.b8`, `.b16`, `.f16`)
/// still occupy a full 32-bit slot because the register file is not byte
/// addressable.
fn slots_for_type(reg_type: &str) -> RegClass {
    // Strip a leading '.' if present so both ".f32" and "f32" parse.
    let t = reg_type.strip_prefix('.').unwrap_or(reg_type);
    match t {
        // Predicate registers live in the condition-code file and do not
        // consume general-purpose registers.
        "pred" => RegClass::Predicate,
        // 64-bit scalar / float types — two 32-bit slots each.
        "b64" | "s64" | "u64" | "f64" => RegClass::General(2),
        // 128-bit (rare, used for vector loads) — four slots.
        "b128" => RegClass::General(4),
        // Everything else (b8/b16/b32, s8..s32, u8..u32, f16/f16x2/f32) — a
        // single 32-bit slot.
        _ => RegClass::General(1),
    }
}

/// Classification of a register declaration for occupancy accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegClass {
    /// Occupies the given number of 32-bit general-purpose register slots.
    General(u32),
    /// A predicate register — does not consume general-purpose registers.
    Predicate,
}

// ---------------------------------------------------------------------------
// PtxRegisterUsage
// ---------------------------------------------------------------------------

/// Register-file usage of a single PTX kernel, derived from its `.reg`
/// directives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PtxRegisterUsage {
    /// Total 32-bit general-purpose register slots consumed per thread.
    general_slots: u32,
    /// Number of declared predicate registers (do not consume GP registers).
    predicate_count: u32,
    /// Number of distinct `.reg` directives parsed (diagnostic).
    directive_count: u32,
}

impl PtxRegisterUsage {
    /// Registers consumed per thread, as `ptxas` would report under the
    /// no-coalescing (`-O0`) lower bound.
    ///
    /// This is the value to pass to the CUDA occupancy model and to
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`.
    #[must_use]
    pub fn registers_per_thread(&self) -> u32 {
        self.general_slots
    }

    /// Number of predicate registers declared by the kernel.
    ///
    /// Predicates are tracked separately because they do not draw from the
    /// general-purpose register file that bounds occupancy.
    #[must_use]
    pub fn predicate_registers(&self) -> u32 {
        self.predicate_count
    }

    /// Number of `.reg` directives that contributed to this usage.
    #[must_use]
    pub fn directive_count(&self) -> u32 {
        self.directive_count
    }

    /// Fold one parsed `.reg` directive into the running totals.
    fn add(&mut self, class: RegClass, count: u32) {
        self.directive_count = self.directive_count.saturating_add(1);
        match class {
            RegClass::General(slots) => {
                self.general_slots = self
                    .general_slots
                    .saturating_add(slots.saturating_mul(count));
            }
            RegClass::Predicate => {
                self.predicate_count = self.predicate_count.saturating_add(count);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing of a single `.reg` line
// ---------------------------------------------------------------------------

/// Parse the vectorised count `<N>` out of a register name token such as
/// `%f<10>` or `%rd<4>`.  A bare name (`%tid`) declares a single register.
fn parse_reg_count(name_token: &str) -> Option<u32> {
    if let Some(open) = name_token.find('<') {
        let close = name_token[open + 1..].find('>')? + open + 1;
        let digits = name_token[open + 1..close].trim();
        digits.parse::<u32>().ok()
    } else {
        // No bracket → one register.
        Some(1)
    }
}

/// Attempt to parse a single line as a `.reg` directive, returning the type
/// class and the declared count.  Returns `None` for any line that is not a
/// register declaration.
///
/// Accepts forms:
/// * `.reg .f32 %f<10>;`
/// * `.reg .b32 %r<5>, %q<3>;`  (comma-separated multi-declaration)
/// * `.reg .pred %p;`           (single register)
fn parse_reg_line(line: &str) -> Option<Vec<(RegClass, u32)>> {
    let trimmed = strip_line_comment(line).trim();
    let rest = trimmed.strip_prefix(".reg")?;
    // The next token must be the type, beginning with a space then '.'.
    let rest = rest.trim_start();
    let mut tokens = rest.split_whitespace();
    let type_token = tokens.next()?;
    if !type_token.starts_with('.') {
        return None;
    }
    let class = slots_for_type(type_token);

    // The remainder (after the type) is the comma-separated name list ending
    // in ';'.  Rebuild it so commas split correctly regardless of spacing.
    let names_part = rest[type_token.len()..].trim().trim_end_matches(';');
    if names_part.is_empty() {
        return None;
    }

    let mut out = Vec::new();
    for name in names_part.split(',') {
        let name = name.trim();
        if name.is_empty() {
            continue;
        }
        let count = parse_reg_count(name)?;
        out.push((class, count));
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Remove a trailing `//` line comment, respecting that `//` inside the body
/// is the only comment form PTX uses on a register line.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

// ---------------------------------------------------------------------------
// Kernel entry extraction
// ---------------------------------------------------------------------------

/// One parsed kernel: its mangled entry name plus its register usage.
#[derive(Debug, Clone)]
pub struct PtxKernel {
    name: String,
    usage: PtxRegisterUsage,
}

impl PtxKernel {
    /// The kernel entry name exactly as it appears after `.entry` in the PTX.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The parsed register-file usage of this kernel.
    #[must_use]
    pub fn usage(&self) -> PtxRegisterUsage {
        self.usage
    }

    /// Convenience accessor for [`PtxRegisterUsage::registers_per_thread`].
    #[must_use]
    pub fn registers_per_thread(&self) -> u32 {
        self.usage.registers_per_thread()
    }
}

/// Extract the entry name that begins at or after `.entry` on `line`.
///
/// Handles both `.visible .entry name(` and `.entry name(` and `.entry name {`.
fn extract_entry_name(after_entry: &str) -> Option<String> {
    let s = after_entry.trim_start();
    // The name runs until the first '(' , whitespace, or '{'.
    let end = s
        .find(|c: char| c == '(' || c == '{' || c.is_whitespace())
        .unwrap_or(s.len());
    let name = &s[..end];
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

// ---------------------------------------------------------------------------
// OccupancyFromPtx
// ---------------------------------------------------------------------------

/// Parses register usage out of a PTX string and wires it into the CPU-side
/// occupancy model.
///
/// Construct with [`OccupancyFromPtx::parse`], then either inspect per-kernel
/// register counts via [`OccupancyFromPtx::kernel`] or compute an occupancy
/// estimate directly with [`OccupancyFromPtx::estimate_for`].
#[derive(Debug, Clone, Default)]
pub struct OccupancyFromPtx {
    kernels: Vec<PtxKernel>,
}

impl OccupancyFromPtx {
    /// Parse all kernels and their `.reg` directives out of a PTX module.
    ///
    /// Register declarations are attributed to the most recently seen
    /// `.entry`.  Directives that appear *before* any `.entry` (module-scope
    /// `.reg`, which PTX permits for globals) are ignored for occupancy
    /// purposes because they do not occupy per-thread registers.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::InvalidValue`] if the PTX contains no `.entry`
    /// directive at all (there is nothing whose occupancy could be modelled).
    pub fn parse(ptx: &str) -> CudaResult<Self> {
        let mut kernels: Vec<PtxKernel> = Vec::new();
        let mut current: Option<usize> = None;
        let mut brace_depth: i32 = 0;

        for raw_line in ptx.lines() {
            let line = strip_line_comment(raw_line);

            // Detect a new `.entry` (with optional `.visible`/`.weak` prefix).
            if let Some(pos) = line.find(".entry") {
                let after = &line[pos + ".entry".len()..];
                if let Some(name) = extract_entry_name(after) {
                    kernels.push(PtxKernel {
                        name,
                        usage: PtxRegisterUsage::default(),
                    });
                    current = Some(kernels.len() - 1);
                }
            }

            // Track brace nesting so we know when a kernel body ends.  A
            // kernel's `.reg` directives only live at brace_depth >= 1 inside
            // its body; once we return to depth 0 the kernel is closed.
            let opens = line.matches('{').count() as i32;
            let closes = line.matches('}').count() as i32;

            // Accumulate `.reg` directives for the current kernel while inside
            // a body.
            if current.is_some() {
                if let Some(decls) = parse_reg_line(line) {
                    if let Some(idx) = current {
                        for (class, count) in decls {
                            kernels[idx].usage.add(class, count);
                        }
                    }
                }
            }

            brace_depth += opens;
            brace_depth -= closes;
            if brace_depth <= 0 {
                brace_depth = 0;
                // Body closed — stop attributing further `.reg` lines to it,
                // but only once we have actually entered a body (opens seen).
                if closes > 0 {
                    current = None;
                }
            }
        }

        if kernels.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self { kernels })
    }

    /// All kernels discovered in the PTX, in source order.
    #[must_use]
    pub fn kernels(&self) -> &[PtxKernel] {
        &self.kernels
    }

    /// Look up a kernel by its entry name.
    #[must_use]
    pub fn kernel(&self, name: &str) -> Option<&PtxKernel> {
        self.kernels.iter().find(|k| k.name == name)
    }

    /// Register usage for the named kernel, or `None` if it is absent.
    #[must_use]
    pub fn register_usage(&self, name: &str) -> Option<PtxRegisterUsage> {
        self.kernel(name).map(PtxKernel::usage)
    }

    /// Estimate occupancy for one kernel on a synthetic device of the given
    /// compute capability, using the register count parsed from the PTX.
    ///
    /// This is the offline equivalent of
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`: the parsed
    /// `.reg` count is substituted for the figure `ptxas` would otherwise
    /// supply.
    ///
    /// # Parameters
    ///
    /// * `kernel_name` — entry name to model.
    /// * `sm_major` / `sm_minor` — target compute capability (e.g. `9, 0`).
    /// * `block_size` — threads per block.
    /// * `shared_memory` — dynamic shared memory per block in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotFound`] if `kernel_name` is not present.
    pub fn estimate_for(
        &self,
        kernel_name: &str,
        sm_major: u32,
        sm_minor: u32,
        block_size: u32,
        shared_memory: u32,
    ) -> CudaResult<OccupancyEstimate> {
        let kernel = self.kernel(kernel_name).ok_or(CudaError::NotFound)?;
        let info = DeviceOccupancyInfo::for_compute_capability(sm_major, sm_minor);
        let calc = OccupancyCalculator::new(info);
        Ok(calc.estimate_occupancy(block_size, kernel.registers_per_thread(), shared_memory))
    }

    /// Estimate occupancy for one kernel against an explicit device
    /// description, for callers that already hold a [`DeviceOccupancyInfo`].
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotFound`] if `kernel_name` is not present.
    pub fn estimate_with_info(
        &self,
        kernel_name: &str,
        info: DeviceOccupancyInfo,
        block_size: u32,
        shared_memory: u32,
    ) -> CudaResult<OccupancyEstimate> {
        let kernel = self.kernel(kernel_name).ok_or(CudaError::NotFound)?;
        let calc = OccupancyCalculator::new(info);
        Ok(calc.estimate_occupancy(block_size, kernel.registers_per_thread(), shared_memory))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy_ext::LimitingFactor;

    const SAXPY: &str = r#"
.version 8.0
.target sm_90
.address_size 64

.visible .entry saxpy(
    .param .u64 saxpy_param_0
)
{
    .reg .f32   %f<8>;
    .reg .b32   %r<4>;
    .reg .pred  %p<2>;
    ret;
}
"#;

    #[test]
    fn parse_reg_count_handles_brackets_and_bare() {
        assert_eq!(parse_reg_count("%f<10>"), Some(10));
        assert_eq!(parse_reg_count("%rd<4>"), Some(4));
        assert_eq!(parse_reg_count("%tid"), Some(1));
        assert_eq!(parse_reg_count("%p<3>"), Some(3));
    }

    #[test]
    fn parse_reg_count_rejects_garbage() {
        assert_eq!(parse_reg_count("%f<abc>"), None);
        assert_eq!(parse_reg_count("%f<"), None);
    }

    #[test]
    fn slots_for_type_matches_register_widths() {
        assert_eq!(slots_for_type(".f32"), RegClass::General(1));
        assert_eq!(slots_for_type(".b32"), RegClass::General(1));
        assert_eq!(slots_for_type(".f64"), RegClass::General(2));
        assert_eq!(slots_for_type(".u64"), RegClass::General(2));
        assert_eq!(slots_for_type(".b128"), RegClass::General(4));
        assert_eq!(slots_for_type(".pred"), RegClass::Predicate);
        // works without the leading dot too
        assert_eq!(slots_for_type("f64"), RegClass::General(2));
    }

    #[test]
    fn parse_reg_line_single_typed() {
        let decls = parse_reg_line(".reg .f32 %f<8>;").expect("should parse");
        assert_eq!(decls, vec![(RegClass::General(1), 8)]);
    }

    #[test]
    fn parse_reg_line_multi_declaration() {
        let decls = parse_reg_line(".reg .b32 %r<4>, %q<3>;").expect("should parse");
        assert_eq!(
            decls,
            vec![(RegClass::General(1), 4), (RegClass::General(1), 3)]
        );
    }

    #[test]
    fn parse_reg_line_strips_comment() {
        let decls = parse_reg_line(".reg .f64 %fd<2>; // doubles").expect("should parse");
        assert_eq!(decls, vec![(RegClass::General(2), 2)]);
    }

    #[test]
    fn parse_reg_line_rejects_non_reg() {
        assert!(parse_reg_line(".visible .entry foo() {").is_none());
        assert!(parse_reg_line("ret;").is_none());
        assert!(parse_reg_line(".version 8.0").is_none());
    }

    #[test]
    fn register_usage_sums_general_slots() {
        let from_ptx = OccupancyFromPtx::parse(SAXPY).expect("valid PTX");
        let usage = from_ptx.register_usage("saxpy").expect("kernel present");
        // 8 f32 (1 slot) + 4 b32 (1 slot) = 12 general; 2 predicates separate.
        assert_eq!(usage.registers_per_thread(), 12);
        assert_eq!(usage.predicate_registers(), 2);
        assert_eq!(usage.directive_count(), 3);
    }

    #[test]
    fn f64_registers_count_double() {
        let ptx = r#"
.visible .entry dgemm() {
    .reg .f64 %fd<10>;
    .reg .b32 %r<2>;
    ret;
}
"#;
        let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid");
        // 10 f64 * 2 slots = 20, + 2 b32 = 22.
        assert_eq!(from_ptx.kernel("dgemm").unwrap().registers_per_thread(), 22);
    }

    #[test]
    fn predicates_do_not_consume_general_registers() {
        let ptx = r#"
.visible .entry only_preds() {
    .reg .pred %p<16>;
    ret;
}
"#;
        let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid");
        let k = from_ptx.kernel("only_preds").unwrap();
        assert_eq!(k.registers_per_thread(), 0);
        assert_eq!(k.usage().predicate_registers(), 16);
    }

    #[test]
    fn multiple_kernels_attributed_separately() {
        let ptx = r#"
.visible .entry ka() {
    .reg .f32 %f<4>;
    ret;
}
.visible .entry kb() {
    .reg .f32 %f<16>;
    .reg .b32 %r<8>;
    ret;
}
"#;
        let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid");
        assert_eq!(from_ptx.kernels().len(), 2);
        assert_eq!(from_ptx.kernel("ka").unwrap().registers_per_thread(), 4);
        assert_eq!(from_ptx.kernel("kb").unwrap().registers_per_thread(), 24);
    }

    #[test]
    fn module_scope_reg_before_entry_ignored() {
        // A `.reg` outside any kernel body must not be attributed to a kernel.
        let ptx = r#"
.reg .b32 %global_dummy<4>;
.visible .entry k() {
    .reg .f32 %f<2>;
    ret;
}
"#;
        let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid");
        assert_eq!(from_ptx.kernel("k").unwrap().registers_per_thread(), 2);
    }

    #[test]
    fn parse_errors_when_no_entry() {
        let ptx = ".version 8.0\n.target sm_70\n";
        assert!(matches!(
            OccupancyFromPtx::parse(ptx),
            Err(CudaError::InvalidValue)
        ));
    }

    #[test]
    fn estimate_for_produces_nonzero_occupancy() {
        let from_ptx = OccupancyFromPtx::parse(SAXPY).expect("valid");
        let est = from_ptx
            .estimate_for("saxpy", 9, 0, 256, 0)
            .expect("kernel present");
        assert!(est.occupancy_ratio > 0.0);
        assert!(est.active_warps_per_sm <= est.max_warps_per_sm);
    }

    #[test]
    fn estimate_for_unknown_kernel_errors() {
        let from_ptx = OccupancyFromPtx::parse(SAXPY).expect("valid");
        assert!(matches!(
            from_ptx.estimate_for("nope", 9, 0, 256, 0),
            Err(CudaError::NotFound)
        ));
    }

    #[test]
    fn high_register_pressure_lowers_occupancy() {
        // A kernel hogging 128 registers/thread should be register-limited on
        // an sm_86 device at a 256-thread block.
        let heavy_ptx = r#"
.visible .entry heavy() {
    .reg .b32 %r<128>;
    ret;
}
"#;
        let heavy = OccupancyFromPtx::parse(heavy_ptx).expect("valid");
        let est_heavy = heavy.estimate_for("heavy", 8, 6, 256, 0).unwrap();
        assert_eq!(est_heavy.limiting_factor, LimitingFactor::Registers);

        // A light kernel (a handful of registers) on the same device / block
        // size must NOT be register-limited.
        let light_ptx = r#"
.visible .entry light() {
    .reg .b32 %r<8>;
    ret;
}
"#;
        let light = OccupancyFromPtx::parse(light_ptx).expect("valid");
        let est_light = light
            .estimate_with_info(
                "light",
                DeviceOccupancyInfo::for_compute_capability(8, 6),
                256,
                0,
            )
            .unwrap();
        assert_ne!(est_light.limiting_factor, LimitingFactor::Registers);
        // And the heavy kernel achieves strictly lower occupancy than light.
        assert!(est_heavy.occupancy_ratio < est_light.occupancy_ratio);
    }

    #[test]
    fn visible_and_plain_entry_both_parsed() {
        let ptx = r#"
.entry plain() {
    .reg .f32 %f<3>;
    ret;
}
"#;
        let from_ptx = OccupancyFromPtx::parse(ptx).expect("valid");
        assert_eq!(from_ptx.kernel("plain").unwrap().registers_per_thread(), 3);
    }
}
