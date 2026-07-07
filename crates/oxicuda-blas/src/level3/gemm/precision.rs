//! Shared PTX precision-conversion helpers for the scalar / SIMT GEMM
//! generators (`simt`, `bandwidth_opt`).
//!
//! These mirror the conversion logic in
//! [`oxicuda_ptx::templates::gemm::GemmTemplate`]: an input element is loaded
//! in its natural precision and converted to/from the accumulator precision.
//! The rules enforced here are dictated by the PTX ISA and validated against
//! `ptxas -arch=sm_86`:
//!
//! * `ld`/`st` have no `.f16`/`.bf16` form — 16-bit floats are moved as raw
//!   `.b16` and reinterpreted by `cvt`.
//! * Narrowing float→float conversions require a rounding modifier; widening
//!   ones forbid it.
//! * There is no direct half↔`f64` `cvt` on pre-Hopper targets, and `bf16` is
//!   designed around `f32` on every architecture, so half↔`f64` conversions are
//!   routed through an `f32` scratch register.

use oxicuda_ptx::ir::PtxType;

/// Returns the PTX `ld`/`st` element type for `precision`. 16-bit floats have
/// no `.f16`/`.bf16` memory-access form and are moved as raw `.b16`.
pub(crate) fn mem_type(precision: PtxType) -> &'static str {
    match precision {
        PtxType::F16 | PtxType::BF16 => ".b16",
        _ => precision.as_ptx_str(),
    }
}

/// Returns `true` when a half↔`f64` conversion needs an intermediate `f32`
/// scratch register bank.
pub(crate) fn needs_f32_scratch(precision: PtxType, accumulator: PtxType) -> bool {
    matches!(precision, PtxType::F16 | PtxType::BF16) && accumulator == PtxType::F64
}

/// Emits the `cvt` instruction(s) converting a freshly loaded input-precision
/// element in register `fin` into the accumulator-precision register `dst`,
/// routing 16-bit floats through the `f32` scratch register `fc` when the
/// accumulator is `F64`.
///
/// Returns the PTX text (one or two indented lines, no trailing newline). The
/// caller must guarantee `precision != accumulator`.
pub(crate) fn convert_to_acc(
    precision: PtxType,
    accumulator: PtxType,
    fin: &str,
    fc: &str,
    dst: &str,
) -> String {
    let in_ty = precision.as_ptx_str();
    match precision {
        PtxType::F16 | PtxType::BF16 => {
            if accumulator == PtxType::F64 {
                format!("    cvt.f32{in_ty} {fc}, {fin};\n    cvt.f64.f32 {dst}, {fc};")
            } else {
                format!("    cvt.f32{in_ty} {dst}, {fin};")
            }
        }
        _ => format!(
            "    {} {dst}, {fin};",
            accumulator.float_cvt_mnemonic(precision)
        ),
    }
}

/// Emits the `cvt` instruction(s) converting the accumulator-precision result
/// register `src` back into the output (input) precision register `fin`, ready
/// for the global store. Mirrors [`convert_to_acc`], routing half↔`f64`
/// narrowing through the `f32` scratch register `fc`.
///
/// The caller must guarantee `precision != accumulator`.
pub(crate) fn convert_from_acc(
    precision: PtxType,
    accumulator: PtxType,
    src: &str,
    fc: &str,
    fin: &str,
) -> String {
    let in_ty = precision.as_ptx_str();
    match precision {
        PtxType::F16 | PtxType::BF16 => {
            if accumulator == PtxType::F64 {
                format!("    cvt.rn.f32.f64 {fc}, {src};\n    cvt.rn{in_ty}.f32 {fin}, {fc};")
            } else {
                format!("    cvt.rn{in_ty}.f32 {fin}, {src};")
            }
        }
        _ => format!(
            "    {} {fin}, {src};",
            precision.float_cvt_mnemonic(accumulator)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_type_half_is_b16() {
        assert_eq!(mem_type(PtxType::F16), ".b16");
        assert_eq!(mem_type(PtxType::BF16), ".b16");
        assert_eq!(mem_type(PtxType::F32), ".f32");
        assert_eq!(mem_type(PtxType::F64), ".f64");
    }

    #[test]
    fn scratch_only_for_half_into_f64() {
        assert!(needs_f32_scratch(PtxType::F16, PtxType::F64));
        assert!(needs_f32_scratch(PtxType::BF16, PtxType::F64));
        assert!(!needs_f32_scratch(PtxType::F16, PtxType::F32));
        assert!(!needs_f32_scratch(PtxType::F32, PtxType::F64));
    }

    #[test]
    fn convert_to_acc_routes_half_via_f32() {
        assert_eq!(
            convert_to_acc(PtxType::F16, PtxType::F32, "%fin1", "%fc1", "%f1"),
            "    cvt.f32.f16 %f1, %fin1;"
        );
        assert_eq!(
            convert_to_acc(PtxType::BF16, PtxType::F64, "%fin1", "%fc1", "%f1"),
            "    cvt.f32.bf16 %fc1, %fin1;\n    cvt.f64.f32 %f1, %fc1;"
        );
        assert_eq!(
            convert_to_acc(PtxType::F32, PtxType::F64, "%fin1", "%fc1", "%f1"),
            "    cvt.f64.f32 %f1, %fin1;"
        );
    }

    #[test]
    fn convert_from_acc_routes_half_via_f32() {
        assert_eq!(
            convert_from_acc(PtxType::F16, PtxType::F32, "%f0", "%fc0", "%fin0"),
            "    cvt.rn.f16.f32 %fin0, %f0;"
        );
        assert_eq!(
            convert_from_acc(PtxType::BF16, PtxType::F64, "%f0", "%fc0", "%fin0"),
            "    cvt.rn.f32.f64 %fc0, %f0;\n    cvt.rn.bf16.f32 %fin0, %fc0;"
        );
        assert_eq!(
            convert_from_acc(PtxType::F32, PtxType::F64, "%f0", "%fc0", "%fin0"),
            "    cvt.rn.f32.f64 %fin0, %f0;"
        );
    }

    // ── ptxas validation for the scalar/SIMT blas GEMM generators ────────────

    /// Locate `ptxas` on PATH (or the well-known CUDA bin dir).
    fn find_ptxas() -> Option<std::path::PathBuf> {
        if let Ok(path) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path) {
                let candidate = dir.join("ptxas");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        let fallback = std::path::PathBuf::from("/usr/local/cuda/bin/ptxas");
        if fallback.is_file() {
            return Some(fallback);
        }
        None
    }

    /// Assembles `ptx` with `ptxas -arch=sm_86`, asserting success.
    fn assert_ptxas_ok(ptxas: &std::path::Path, tag: &str, ptx: &str) {
        let mut ptx_path = std::env::temp_dir();
        ptx_path.push(format!(
            "oxicuda_blas_gemm_{tag}_{}.ptx",
            std::process::id()
        ));
        std::fs::write(&ptx_path, ptx).expect("write PTX to temp file");

        let output = std::process::Command::new(ptxas)
            .arg("-arch=sm_86")
            .arg(&ptx_path)
            .arg("-o")
            .arg("/dev/null")
            .output()
            .expect("invoke ptxas");

        let _ = std::fs::remove_file(&ptx_path);

        assert!(
            output.status.success(),
            "ptxas rejected {tag} GEMM PTX:\n{}\n--- PTX ---\n{ptx}",
            String::from_utf8_lossy(&output.stderr),
        );
    }

    /// Every scalar/SIMT GEMM generator must emit PTX that `ptxas -arch=sm_86`
    /// accepts for the precisions it admits — most importantly F64, the path
    /// that previously emitted `.f32` register banks with `.f64` instructions.
    #[test]
    fn blas_scalar_gemm_generators_assemble_for_sm86() {
        use oxicuda_ptx::arch::SmVersion;

        use super::super::bandwidth_opt::{
            BandwidthGemmConfig, BandwidthPrecision, BandwidthStrategy, generate_bandwidth_gemm_ptx,
        };
        use super::super::simt::SimtGemmBuilder;
        use super::super::splitk::generate_splitk_reduction_kernel;
        use crate::types::Transpose;

        let Some(ptxas) = find_ptxas() else {
            println!("skipping: ptxas not found on PATH");
            return;
        };

        // SIMT: both same-precision (F32/F64) and mixed (F16/F32) inputs.
        for (precision, accumulator) in [
            (PtxType::F32, PtxType::F32),
            (PtxType::F64, PtxType::F64),
            (PtxType::F16, PtxType::F32),
        ] {
            let ptx = SimtGemmBuilder::new(
                SmVersion::Sm86,
                precision,
                accumulator,
                Transpose::NoTrans,
                Transpose::NoTrans,
                None,
            )
            .generate()
            .expect("SIMT GEMM should generate");
            assert_ptxas_ok(
                &ptxas,
                &format!(
                    "simt_{}_{}",
                    precision.as_ptx_str().trim_start_matches('.'),
                    accumulator.as_ptx_str().trim_start_matches('.')
                ),
                &ptx,
            );
        }

        // Bandwidth-optimised: F32, F64, and F16 (f32 accumulator) precisions.
        for prec in [
            BandwidthPrecision::F32,
            BandwidthPrecision::F64,
            BandwidthPrecision::F16,
        ] {
            let cfg = BandwidthGemmConfig {
                m: 256,
                n: 256,
                k: 16,
                sm_version: SmVersion::Sm86,
                precision: prec,
                strategy: BandwidthStrategy::Auto,
            };
            let ptx = generate_bandwidth_gemm_ptx(&cfg).expect("bandwidth GEMM should generate");
            assert_ptxas_ok(&ptxas, &format!("bw_{prec:?}"), &ptx);
        }

        // Split-K reduction kernel: F32 and F64 accumulators.
        for acc in [PtxType::F32, PtxType::F64] {
            let (_, ptx) = generate_splitk_reduction_kernel(SmVersion::Sm86, acc, 4)
                .expect("split-K reduction should generate");
            assert_ptxas_ok(
                &ptxas,
                &format!("splitk_{}", acc.as_ptx_str().trim_start_matches('.')),
                &ptx,
            );
        }
    }
}
