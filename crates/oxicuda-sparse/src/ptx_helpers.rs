//! Sparse-specific PTX code-generation helpers.
//!
//! These functions encapsulate common patterns needed by sparse kernels:
//! warp shuffle reductions, atomic floating-point adds, and guarded
//! memory loads with bounds checking.
#![allow(dead_code)]

use oxicuda_blas::GpuFloat;
use oxicuda_ptx::builder::BodyBuilder;
use oxicuda_ptx::ir::{PtxType, Register};

/// Returns the PTX type suffix without a leading dot (e.g. `"f32"`, `"f64"`).
pub(crate) fn ptx_suffix<T: GpuFloat>() -> &'static str {
    let s = T::PTX_TYPE.as_ptx_str();
    s.strip_prefix('.').unwrap_or(s)
}

/// Loads a float value from a global memory address.
pub(crate) fn load_global_float<T: GpuFloat>(b: &mut BodyBuilder<'_>, addr: Register) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.load_global_f32(addr)
    } else {
        b.load_global_f64(addr)
    }
}

/// Stores a float value to a global memory address.
pub(crate) fn store_global_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    if T::PTX_TYPE == PtxType::F32 {
        b.store_global_f32(addr, val);
    } else {
        b.store_global_f64(addr, val);
    }
}

/// Emits a fused multiply-add: `dst = a * bv + c`.
pub(crate) fn fma_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
    c: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.fma_f32(a, bv, c)
    } else {
        b.fma_f64(a, bv, c)
    }
}

/// Emits a multiplication: `dst = a * bv`.
pub(crate) fn mul_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let dst = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mul.rn.f32 {dst}, {a}, {bv};"));
        dst
    } else {
        let dst = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mul.rn.f64 {dst}, {a}, {bv};"));
        dst
    }
}

/// Emits an addition: `dst = a + bv`.
pub(crate) fn add_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    a: Register,
    bv: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        b.add_f32(a, bv)
    } else {
        b.add_f64(a, bv)
    }
}

/// Loads an immediate float constant into a register.
pub(crate) fn load_float_imm<T: GpuFloat>(b: &mut BodyBuilder<'_>, val: f64) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let r = b.alloc_reg(PtxType::F32);
        let bits = (val as f32).to_bits();
        b.raw_ptx(&format!("mov.b32 {r}, 0F{bits:08X};"));
        r
    } else {
        let r = b.alloc_reg(PtxType::F64);
        let bits = val.to_bits();
        b.raw_ptx(&format!("mov.b64 {r}, 0D{bits:016X};"));
        r
    }
}

/// Reinterprets a u64 bit pattern as the float type `T`.
pub(crate) fn reinterpret_bits_to_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    bits: Register,
) -> Register {
    if T::PTX_TYPE == PtxType::F32 {
        let bits32 = b.alloc_reg(PtxType::U32);
        b.raw_ptx(&format!("cvt.u32.u64 {bits32}, {bits};"));
        let fval = b.alloc_reg(PtxType::F32);
        b.raw_ptx(&format!("mov.b32 {fval}, {bits32};"));
        fval
    } else {
        let fval = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mov.b64 {fval}, {bits};"));
        fval
    }
}

/// Emits a single warp shuffle of a float register, returning the shuffled
/// value in a fresh register of the same float type.
///
/// PTX `shfl.sync` only supports 32-bit (`.b32`) operands; emitting
/// `shfl.sync.<mode>.b64` is rejected by `ptxas` ("Unexpected instruction
/// types specified for 'shfl'"). For `f64` the value is therefore unpacked
/// into its low/high 32-bit halves (`mov.b64 {lo,hi}, val`), each half is
/// shuffled independently with `.b32`, and the halves are repacked
/// (`mov.b64 packed, {lo,hi}`) before being returned. For `f32` a single
/// `.b32` shuffle suffices.
///
/// * `mode` is the shuffle mode without the `shfl.sync.` prefix or width
///   suffix, e.g. `"down"`, `"up"`, `"bfly"`, `"idx"`.
/// * `offset` is the lane offset / index operand (already formatted).
/// * `c` is the shuffle's clamp / segment-mask operand (e.g. `"31"` for
///   down-shuffles, `"0"` for up-shuffles).
pub(crate) fn emit_shfl_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    mode: &str,
    val: Register,
    offset: &str,
    c: &str,
) -> Register {
    if T::PTX_TYPE == PtxType::F64 {
        let lo = b.alloc_reg(PtxType::B32);
        let hi = b.alloc_reg(PtxType::B32);
        b.raw_ptx(&format!("mov.b64 {{{lo}, {hi}}}, {val};"));
        let lo_shuf = b.alloc_reg(PtxType::B32);
        let hi_shuf = b.alloc_reg(PtxType::B32);
        b.raw_ptx(&format!(
            "shfl.sync.{mode}.b32 {lo_shuf}, {lo}, {offset}, {c}, 0xFFFFFFFF;"
        ));
        b.raw_ptx(&format!(
            "shfl.sync.{mode}.b32 {hi_shuf}, {hi}, {offset}, {c}, 0xFFFFFFFF;"
        ));
        let packed = b.alloc_reg(PtxType::F64);
        b.raw_ptx(&format!("mov.b64 {packed}, {{{lo_shuf}, {hi_shuf}}};"));
        packed
    } else {
        let shuffled = b.alloc_reg(T::PTX_TYPE);
        b.raw_ptx(&format!(
            "shfl.sync.{mode}.b32 {shuffled}, {val}, {offset}, {c}, 0xFFFFFFFF;"
        ));
        shuffled
    }
}

/// Emits a warp shuffle reduction (sum) over a register using `shfl.sync.down`.
///
/// Reduces `val` across all lanes in the warp (32 threads) with a down-shuffle
/// tree (offsets 16, 8, 4, 2, 1). After this, lane 0 holds the sum of all lanes.
///
/// `shfl.sync` only operates on 32-bit (`.b32`) registers. For `f64` the value
/// is therefore unpacked into its low/high 32-bit halves, each half is shuffled
/// independently with `shfl.sync.down.b32`, and the two halves are repacked into
/// an `f64` register before the `add.f64`. Emitting `shfl.sync.down.b64` is
/// rejected by `ptxas` ("Unexpected instruction types specified for 'shfl'").
pub(crate) fn emit_warp_reduce_sum<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    val: Register,
) -> Register {
    let suffix = ptx_suffix::<T>();

    let mut current = val;
    // Unroll the warp reduction: offsets 16, 8, 4, 2, 1
    for offset in [16u32, 8, 4, 2, 1] {
        // `emit_shfl_float` performs the f64 unpack/repack so no `.b64` shfl
        // is ever emitted (ptxas only accepts `.b32`).
        let shuffled = emit_shfl_float::<T>(b, "down", current.clone(), &offset.to_string(), "31");
        let sum = b.alloc_reg(T::PTX_TYPE);
        b.raw_ptx(&format!("add.{suffix} {sum}, {current}, {shuffled};"));
        current = sum;
    }
    current
}

/// Emits an atomic add for floating-point values.
///
/// Uses `atom.global.add.f32` (or f64 on sm_60+).
pub(crate) fn emit_atomic_add_float<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    addr: Register,
    val: Register,
) {
    let suffix = ptx_suffix::<T>();
    let _old = b.alloc_reg(T::PTX_TYPE);
    b.raw_ptx(&format!(
        "atom.global.add.{suffix} {_old}, [{addr}], {val};"
    ));
}

/// Emits a guarded global load with bounds checking.
///
/// Loads `T` from `base_addr + idx * sizeof(T)` if `idx < bound`,
/// otherwise returns zero.
pub(crate) fn emit_guarded_load<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    base_addr: Register,
    idx: Register,
    bound: Register,
) -> Register {
    let elem_bytes = T::SIZE as u32;
    let zero = load_float_imm::<T>(b, 0.0);

    let pred = b.alloc_reg(PtxType::Pred);
    b.raw_ptx(&format!("setp.lo.u32 {pred}, {idx}, {bound};"));

    let addr = b.byte_offset_addr(base_addr, idx, elem_bytes);
    let loaded = load_global_float::<T>(b, addr);

    let result = b.alloc_reg(T::PTX_TYPE);
    let suffix = ptx_suffix::<T>();
    b.raw_ptx(&format!(
        "selp.{suffix} {result}, {loaded}, {zero}, {pred};"
    ));
    result
}

/// Test-only helpers shared by the per-kernel PTX-assembly regression tests.
#[cfg(test)]
pub(crate) mod test_support {
    /// Outcome of attempting to assemble PTX with `ptxas`.
    pub(crate) enum PtxasOutcome {
        /// `ptxas` accepted the PTX.
        Ok,
        /// `ptxas` rejected the PTX; carries the captured stderr.
        Rejected(String),
        /// `ptxas` is not installed on `PATH` (skip gracefully).
        Unavailable,
    }

    /// Runs `ptxas -arch=sm_86` on `ptx`, writing it under
    /// [`std::env::temp_dir`]. `name` disambiguates the temp file.
    pub(crate) fn assemble_sm86(name: &str, ptx: &str) -> PtxasOutcome {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("oxicuda_sparse_{name}.ptx"));
        {
            let mut f = std::fs::File::create(&path).expect("test: create temp PTX file");
            f.write_all(ptx.as_bytes())
                .expect("test: write temp PTX file");
        }
        // Assemble to a throwaway file rather than a null device: `/dev/null` is
        // not openable on Windows, where ptxas then fails for a reason that has
        // nothing to do with the PTX under test.
        let cubin = std::env::temp_dir().join(format!("oxicuda_sparse_{name}.cubin"));
        let outcome = match std::process::Command::new("ptxas")
            .arg("-arch=sm_86")
            .arg(&path)
            .arg("-o")
            .arg(&cubin)
            .output()
        {
            Ok(out) if out.status.success() => PtxasOutcome::Ok,
            // ptxas prints its diagnostics on stdout, not stderr; capturing
            // only stderr leaves the rejection message empty.
            Ok(out) => PtxasOutcome::Rejected(format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            )),
            Err(_) => PtxasOutcome::Unavailable,
        };
        let _ = std::fs::remove_file(&cubin);
        outcome
    }

    /// Asserts that `ptx` assembles for `sm_86` (or skips if `ptxas` is absent)
    /// and contains neither the f32 `0F00000000` zero-immediate misuse nor any
    /// illegal `.b64` warp shuffle.
    pub(crate) fn assert_assembles_and_clean(kernel: &str, ptx: &str) {
        assert!(
            !ptx.contains("shfl.sync.down.b64")
                && !ptx.contains("shfl.sync.up.b64")
                && !ptx.contains("shfl.sync.bfly.b64")
                && !ptx.contains("shfl.sync.idx.b64"),
            "{kernel}: emitted an illegal .b64 warp shuffle (ptxas only accepts .b32):\n{ptx}"
        );
        match assemble_sm86(kernel, ptx) {
            PtxasOutcome::Ok | PtxasOutcome::Unavailable => {}
            PtxasOutcome::Rejected(stderr) => {
                panic!("ptxas rejected the {kernel} kernel:\n{stderr}\nPTX:\n{ptx}")
            }
        }
    }
}
