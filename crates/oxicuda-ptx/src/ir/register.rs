//! PTX register model and allocation.
//!
//! PTX uses a virtual (infinite) register model. The `ptxas` assembler maps
//! virtual registers to physical registers during compilation. This module
//! provides [`Register`] as a named, typed register and [`RegisterAllocator`]
//! for generating unique register names following PTX naming conventions.

use std::collections::HashMap;
use std::fmt;

use super::types::PtxType;
use crate::error::PtxGenError;

/// A named PTX register with an associated type.
///
/// Register names follow PTX conventions: `%f0` for floats, `%r0` for 32-bit
/// integers, `%rd0` for 64-bit integers/addresses, `%p0` for predicates.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Register {
    /// The register name (e.g., `"%f0"`, `"%r3"`).
    pub name: String,
    /// The PTX type of values held in this register.
    pub ty: PtxType,
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Register allocator for PTX code generation.
///
/// PTX uses an infinite register model — `ptxas` maps virtual registers to
/// physical registers. This allocator generates sequentially numbered register
/// names grouped by type class, and can emit the `.reg` declarations needed
/// at the top of a PTX function body.
///
/// # Register naming conventions
///
/// | Type class              | Prefix | Example |
/// |-------------------------|--------|---------|
/// | Predicate               | `%p`   | `%p0`   |
/// | F16, BF16, F32, F64     | `%f`   | `%f0`   |
/// | B64, U64, S64           | `%rd`  | `%rd0`  |
/// | Everything else (32-bit)| `%r`   | `%r0`   |
pub struct RegisterAllocator {
    /// Per-prefix list of each allocated register's *size class*
    /// ([`PtxType::reg_type`]), indexed by the register's numeric id. The
    /// vector length is the next free id for that prefix, so it doubles as the
    /// name counter. Storing the per-index size class (rather than a single
    /// per-prefix type) lets [`emit_declarations`](Self::emit_declarations)
    /// declare a *heterogeneous* bank — e.g. an `f64` kernel that also needs
    /// `f32` SFU scratch (`ex2.approx.f32` has no `f64` form), which mixes
    /// `.b64` and `.b32` registers under the shared `%f` prefix — correctly,
    /// instead of forcing one size on every register and having ptxas reject
    /// the size-mismatched uses.
    reg_types: HashMap<&'static str, Vec<PtxType>>,
    /// Explicitly named registers (e.g., `%f_x`) declared via [`Self::declare_named`].
    named_registers: Vec<(String, PtxType)>,
}

impl RegisterAllocator {
    /// Creates a new register allocator with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            reg_types: HashMap::new(),
            named_registers: Vec::new(),
        }
    }

    /// Allocates a fresh register of the given type.
    ///
    /// Returns a [`Register`] with a unique name following PTX naming conventions.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxicuda_ptx::ir::{RegisterAllocator, PtxType};
    ///
    /// let mut alloc = RegisterAllocator::new();
    /// let r0 = alloc.alloc(PtxType::F32);
    /// assert_eq!(r0.name, "%f0");
    /// let r1 = alloc.alloc(PtxType::F32);
    /// assert_eq!(r1.name, "%f1");
    /// let ri = alloc.alloc(PtxType::U32);
    /// assert_eq!(ri.name, "%r0");
    /// ```
    pub fn alloc(&mut self, ty: PtxType) -> Register {
        let prefix = Self::prefix_for(ty);
        let bank = self.reg_types.entry(prefix).or_default();
        // The register's numeric id is its position in the bank.
        let idx = bank.len();
        // Record the register's size class (b16/b32/b64/pred) at this index so
        // declarations can be emitted per-register when the bank is mixed.
        bank.push(ty.reg_type());

        Register {
            name: format!("%{prefix}{idx}"),
            ty,
        }
    }

    /// Allocates a group of registers of the same type.
    ///
    /// Returns `count` sequentially numbered registers.
    pub fn alloc_group(&mut self, ty: PtxType, count: u32) -> Vec<Register> {
        (0..count).map(|_| self.alloc(ty)).collect()
    }

    /// Declares a named register for use in raw PTX.
    ///
    /// Unlike [`alloc`](Self::alloc), this accepts an arbitrary register name
    /// (e.g., `%f_x`, `%rd_off`) and ensures it appears in the `.reg`
    /// declarations emitted by [`emit_declarations`](Self::emit_declarations).
    pub fn declare_named(&mut self, name: &str, ty: PtxType) {
        if !self.named_registers.iter().any(|(n, _)| n == name) {
            self.named_registers.push((name.to_string(), ty));
        }
    }

    /// Validates all explicitly-declared named registers.
    ///
    /// Rejects a named register that would produce a duplicate or illegal
    /// `.reg` declaration:
    /// - a name that is not a syntactically valid `%`-prefixed PTX identifier;
    /// - a name matching the reserved allocator pattern `%(p|f|rd|r)<digits>`,
    ///   which collides with the compact range declarations
    ///   (`.reg .f32 %f<N>;`) that cover `%f0..%f{N-1}` etc.; or
    /// - a name that shadows a PTX special register (`%tid`, `%laneid`,
    ///   `%clock64`, `%envreg0`, ...).
    ///
    /// # Errors
    ///
    /// Returns [`PtxGenError::RegisterError`] describing the first offending
    /// named register.
    pub fn validate_named(&self) -> Result<(), PtxGenError> {
        for (name, _) in &self.named_registers {
            if !is_valid_reg_ident(name) {
                return Err(PtxGenError::RegisterError(format!(
                    "named register {name:?} is not a valid PTX register identifier"
                )));
            }
            if is_reserved_allocator_name(name) {
                return Err(PtxGenError::RegisterError(format!(
                    "named register {name:?} collides with an allocator-generated register \
                     (reserved pattern %(p|f|rd|r)<digits>)"
                )));
            }
            if is_ptx_special_register(name) {
                return Err(PtxGenError::RegisterError(format!(
                    "named register {name:?} collides with a PTX special register"
                )));
            }
        }
        Ok(())
    }

    /// Emits `.reg` declaration lines for all allocated register types.
    ///
    /// Each declaration uses the PTX range syntax (e.g., `.reg .f32 %f<4>;`)
    /// which declares registers `%f0` through `%f3`.
    ///
    /// # Returns
    ///
    /// A vector of declaration strings, one per register prefix used.
    #[must_use]
    pub fn emit_declarations(&self) -> Vec<String> {
        let mut declarations = Vec::new();

        for (prefix, sizes) in &self.reg_types {
            let Some(first) = sizes.first() else {
                continue;
            };
            // `sizes` already holds each register's size class (reg_type).
            if sizes.iter().all(|s| s == first) {
                // Homogeneous bank: one compact range declaration.
                declarations.push(format!(
                    ".reg {} %{prefix}<{}>;",
                    first.as_ptx_str(),
                    sizes.len()
                ));
            } else {
                // Heterogeneous bank (only the float `%f` bank can mix b16/b32/
                // b64 — e.g. an f64 kernel with f32 SFU scratch). A single range
                // declaration would force one size on every register and ptxas
                // would reject the size-mismatched uses, so declare each
                // register with its own size class.
                for (i, size) in sizes.iter().enumerate() {
                    declarations.push(format!(".reg {} %{prefix}{i};", size.as_ptx_str()));
                }
            }
        }

        // Emit named register declarations (e.g., `.reg .b64 %rd_off;`).
        for (name, ty) in &self.named_registers {
            declarations.push(format!(".reg {} {name};", ty.reg_type().as_ptx_str()));
        }

        declarations.sort();
        declarations
    }

    /// Returns the register name prefix for a given PTX type.
    const fn prefix_for(ty: PtxType) -> &'static str {
        match ty {
            PtxType::Pred => "p",
            PtxType::F16
            | PtxType::F16x2
            | PtxType::BF16
            | PtxType::BF16x2
            | PtxType::F32
            | PtxType::F64 => "f",
            PtxType::B64 | PtxType::U64 | PtxType::S64 => "rd",
            _ => "r",
        }
    }
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if `name` is a syntactically valid `%`-prefixed PTX register
/// identifier: `%[A-Za-z_][A-Za-z0-9_$]*`.
fn is_valid_reg_ident(name: &str) -> bool {
    let mut chars = name.chars();
    if chars.next() != Some('%') {
        return false;
    }
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
}

/// Returns `true` if `name` matches the reserved allocator pattern
/// `%(p|f|rd|r)<digits>` used by [`RegisterAllocator::alloc`].
fn is_reserved_allocator_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('%') else {
        return false;
    };
    // Try each allocator prefix; the numeric suffix must be non-empty digits.
    for prefix in ["rd", "p", "f", "r"] {
        if let Some(digits) = rest.strip_prefix(prefix) {
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if `name` shadows a PTX special (built-in) register.
fn is_ptx_special_register(name: &str) -> bool {
    const SPECIAL: &[&str] = &[
        "%tid",
        "%ntid",
        "%ctaid",
        "%nctaid",
        "%laneid",
        "%warpid",
        "%nwarpid",
        "%smid",
        "%nsmid",
        "%gridid",
        "%lanemask_eq",
        "%lanemask_le",
        "%lanemask_lt",
        "%lanemask_ge",
        "%lanemask_gt",
        "%clock",
        "%clock_hi",
        "%clock64",
        "%globaltimer",
        "%globaltimer_lo",
        "%globaltimer_hi",
        "%dynamic_smem_size",
        "%total_smem_size",
        "%aggr_smem_size",
        "%current_graph_exec",
    ];
    if SPECIAL.contains(&name) {
        return true;
    }
    // The reserved shared-memory-offset family has arbitrary suffixes.
    if name.starts_with("%reserved_smem_offset_") {
        return true;
    }
    // Numeric families: %envreg0..31, %pm0..7.
    for prefix in ["%envreg", "%pm"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_float_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::F32);
        let r1 = alloc.alloc(PtxType::F32);
        let r2 = alloc.alloc(PtxType::F64);
        assert_eq!(r0.name, "%f0");
        assert_eq!(r1.name, "%f1");
        assert_eq!(r2.name, "%f2");
        assert_eq!(r0.ty, PtxType::F32);
        assert_eq!(r2.ty, PtxType::F64);
    }

    #[test]
    fn alloc_integer_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::U32);
        let r1 = alloc.alloc(PtxType::S32);
        assert_eq!(r0.name, "%r0");
        assert_eq!(r1.name, "%r1");
    }

    #[test]
    fn alloc_64bit_registers() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::U64);
        let r1 = alloc.alloc(PtxType::S64);
        let r2 = alloc.alloc(PtxType::B64);
        assert_eq!(r0.name, "%rd0");
        assert_eq!(r1.name, "%rd1");
        assert_eq!(r2.name, "%rd2");
    }

    #[test]
    fn alloc_predicate_registers() {
        let mut alloc = RegisterAllocator::new();
        let p0 = alloc.alloc(PtxType::Pred);
        let p1 = alloc.alloc(PtxType::Pred);
        assert_eq!(p0.name, "%p0");
        assert_eq!(p1.name, "%p1");
    }

    #[test]
    fn alloc_group() {
        let mut alloc = RegisterAllocator::new();
        let regs = alloc.alloc_group(PtxType::F32, 4);
        assert_eq!(regs.len(), 4);
        assert_eq!(regs[0].name, "%f0");
        assert_eq!(regs[3].name, "%f3");
    }

    #[test]
    fn emit_declarations_sorted() {
        let mut alloc = RegisterAllocator::new();
        alloc.alloc(PtxType::F32);
        alloc.alloc(PtxType::F32);
        alloc.alloc(PtxType::U32);
        alloc.alloc(PtxType::Pred);
        alloc.alloc(PtxType::U64);

        let decls = alloc.emit_declarations();
        assert_eq!(decls.len(), 4);
        // Declarations are sorted alphabetically by the full string.
        // Check that all expected declarations are present.
        let joined = decls.join("\n");
        assert!(joined.contains("%f<2>"), "missing f decl: {joined}");
        assert!(joined.contains("%p<1>"), "missing p decl: {joined}");
        assert!(joined.contains("%r<1>"), "missing r decl: {joined}");
        assert!(joined.contains("%rd<1>"), "missing rd decl: {joined}");
        // Verify sorting: each decl should be <= the next.
        for pair in decls.windows(2) {
            assert!(pair[0] <= pair[1], "declarations not sorted: {decls:?}");
        }
    }

    #[test]
    fn register_display() {
        let r = Register {
            name: "%f0".to_string(),
            ty: PtxType::F32,
        };
        assert_eq!(format!("{r}"), "%f0");
    }

    #[test]
    fn validate_named_accepts_custom_names() {
        let mut alloc = RegisterAllocator::new();
        alloc.declare_named("%f_x", PtxType::F32);
        alloc.declare_named("%rd_off", PtxType::B64);
        alloc.declare_named("%p_ge", PtxType::Pred);
        alloc.declare_named("%fd_acc", PtxType::F64);
        assert!(alloc.validate_named().is_ok());
    }

    #[test]
    fn validate_named_rejects_allocator_collision() {
        let mut alloc = RegisterAllocator::new();
        alloc.declare_named("%f0", PtxType::F32); // collides with %f<N> range
        assert!(alloc.validate_named().is_err());

        let mut alloc2 = RegisterAllocator::new();
        alloc2.declare_named("%rd7", PtxType::B64); // collides with %rd<N>
        assert!(alloc2.validate_named().is_err());
    }

    #[test]
    fn validate_named_rejects_special_registers() {
        for special in ["%clock64", "%laneid", "%envreg3", "%pm0", "%tid"] {
            let mut alloc = RegisterAllocator::new();
            alloc.declare_named(special, PtxType::U32);
            assert!(
                alloc.validate_named().is_err(),
                "{special} must be rejected as a special register"
            );
        }
    }

    #[test]
    fn validate_named_rejects_invalid_identifier() {
        let mut alloc = RegisterAllocator::new();
        alloc.declare_named("no_percent", PtxType::U32);
        assert!(alloc.validate_named().is_err());

        let mut alloc2 = RegisterAllocator::new();
        alloc2.declare_named("%1bad", PtxType::U32); // first char is a digit
        assert!(alloc2.validate_named().is_err());
    }

    #[test]
    fn helper_reserved_and_special_classification() {
        assert!(is_reserved_allocator_name("%r3"));
        assert!(is_reserved_allocator_name("%rd0"));
        assert!(is_reserved_allocator_name("%f12"));
        assert!(is_reserved_allocator_name("%p1"));
        assert!(!is_reserved_allocator_name("%f_x"));
        assert!(!is_reserved_allocator_name("%rd_off"));
        assert!(is_ptx_special_register("%globaltimer"));
        assert!(is_ptx_special_register("%reserved_smem_offset_begin"));
        assert!(!is_ptx_special_register("%my_reg"));
        assert!(is_valid_reg_ident("%f_x"));
        assert!(!is_valid_reg_ident("%f.x"));
    }
}
