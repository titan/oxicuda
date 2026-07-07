//! Stopgap relocation of misplaced PTX performance-tuning directives.
//!
//! # Why this exists
//!
//! Several `oxicuda-ptx` code generators emit the `.maxntid` performance-tuning
//! directive *inside* the kernel body — immediately after the opening `{`:
//!
//! ```text
//! .visible .entry reduce_sum_f32_bs256( .param ... )
//! {
//!     .maxntid 256, 1, 1;     // <-- rejected here
//!     .reg .b32 %r<16>;
//! ```
//!
//! `ptxas` only accepts performance-tuning directives (`.maxntid`, `.reqntid`,
//! `.minnctapersm`, `.maxnctapersm`, `.maxnreg`) **between** the entry's
//! parameter-list `)` and the body `{`. With the directive in the body every
//! such module is rejected at JIT time with
//! `Parsing error near '.maxntid': syntax error`, so the reduction / softmax
//! kernels never load on the device.
//!
//! The affected generators are `ReductionTemplate`, `PerAxisReductionTemplate`,
//! `SoftmaxTemplate` (block path), and `generate_multi_block_softmax_ptx`. The
//! warp-softmax path, the elementwise `Scale` template, and every `KernelBuilder`
//! kernel place the directive correctly and are unaffected.
//!
//! [`relocate_perf_directives`] moves any in-body directive to the only spot
//! `ptxas` accepts, so the production reduction launchers below produce loadable
//! modules. This is a BLAS-layer workaround; the proper fix is in `oxicuda-ptx`
//! (move the `writeln!(... ".maxntid ...")` call to before the body `{`, with no
//! trailing semicolon). The transform is idempotent — PTX whose directives are
//! already correctly placed is returned unchanged.

/// PTX performance-tuning directives that must appear between the entry's
/// parameter-list `)` and the kernel body `{` rather than inside the body.
const PERF_DIRECTIVES: [&str; 5] = [
    ".maxntid",
    ".reqntid",
    ".minnctapersm",
    ".maxnctapersm",
    ".maxnreg",
];

/// Relocates any performance-tuning directive emitted as the first statement of
/// a kernel body to immediately before that body's opening `{`, stripping the
/// trailing `;` (the directive takes no semicolon in directive position).
///
/// Idempotent: a directive already sitting between `)` and `{` is left as-is, so
/// applying this twice — or to PTX from a fixed generator — is a no-op.
pub(crate) fn relocate_perf_directives(ptx: &str) -> String {
    let mut out: Vec<String> = Vec::with_capacity(ptx.split('\n').count());
    for line in ptx.split('\n') {
        let is_perf = {
            let trimmed = line.trim_start();
            PERF_DIRECTIVES.iter().any(|d| trimmed.starts_with(d))
        };
        // The directive immediately follows the body `{`: hoist it above the
        // brace and drop the trailing `;`. `out.last() == Some("{")` guarantees
        // `pop()` yields the brace, but match it explicitly to stay panic-free.
        if is_perf && out.last().map(|l| l.trim()) == Some("{") {
            if let Some(brace) = out.pop() {
                out.push(line.trim_end().trim_end_matches(';').to_string());
                out.push(brace);
                continue;
            }
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const BROKEN: &str = "\
.visible .entry reduce_sum_f32_bs256(
    .param .u64 %param_input,
    .param .u32 %param_n
)
{
    .maxntid 256, 1, 1;
    .reg .b32 %r<16>;
    ret;
}
";

    #[test]
    fn relocates_maxntid_above_brace() {
        let fixed = relocate_perf_directives(BROKEN);
        let lines: Vec<&str> = fixed.split('\n').collect();
        let dir = lines
            .iter()
            .position(|l| l.trim_start().starts_with(".maxntid"))
            .expect("directive present");
        let brace = lines
            .iter()
            .position(|l| l.trim() == "{")
            .expect("brace present");
        assert!(dir < brace, "directive must precede the body brace");
        // Semicolon dropped in directive position.
        assert!(!lines[dir].trim_end().ends_with(';'));
        // No performance directive remains inside the body.
        assert!(
            !fixed.contains("{\n    .maxntid"),
            "no in-body directive should remain"
        );
    }

    #[test]
    fn preserves_directive_arguments_and_entry() {
        let fixed = relocate_perf_directives(BROKEN);
        assert!(fixed.contains(".maxntid 256, 1, 1"));
        assert!(fixed.contains(".visible .entry reduce_sum_f32_bs256("));
        assert!(fixed.trim_end().ends_with('}'));
    }

    #[test]
    fn idempotent_on_already_fixed_ptx() {
        let once = relocate_perf_directives(BROKEN);
        let twice = relocate_perf_directives(&once);
        assert_eq!(once, twice, "relocation must be idempotent");
    }

    #[test]
    fn leaves_clean_ptx_untouched() {
        let clean = "\
.visible .entry warp(
    .param .u64 %p
)
{
    .reg .b32 %r<4>;
    ret;
}
";
        assert_eq!(relocate_perf_directives(clean), clean);
    }

    #[test]
    fn preserves_trailing_newline() {
        assert!(relocate_perf_directives(BROKEN).ends_with('\n'));
        let no_nl = BROKEN.trim_end();
        assert!(!relocate_perf_directives(no_nl).ends_with('\n'));
    }
}
