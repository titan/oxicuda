//! On-device numeric validation for the BLAS elementwise + complex kernels.
//!
//! A convergence audit confirmed these kernels are launched in production but
//! were never numerically validated on device (only GEMM was). This module
//! closes that gap: every test drives the *production* op (which JITs and
//! launches the real PTX on the live GPU), copies the result back, and asserts
//! equivalence to an independent CPU oracle that re-derives the math.
//!
//! Covered surfaces:
//! * unary elementwise (`elementwise/unary.rs`) — every activation / math op,
//! * binary elementwise (`elementwise/binary.rs`) — arithmetic, comparison,
//!   fuzzy-logic, and fused ops,
//! * complex GEMM / GEMV (`complex_gemm.rs`) — all transpose + conjugate paths
//!   with non-trivial complex `alpha`/`beta`, `beta = 0`, strides, and a
//!   non-square (true-matmul) shape.
//!
//! Tolerances: exact arithmetic (add/mul/relu/abs/sqrt.rn/cmp/…) is checked at
//! `1e-4` rel/abs; ops built on the SFU base-2 approximations
//! (`ex2.approx`/`lg2.approx`/`rcp.approx`/`rsqrt.approx`: exp, log, sigmoid,
//! tanh, gelu, silu, softplus, rsqrt, pow) use a looser `5e-3`. Complex f64
//! uses `1e-10`. Every test skips cleanly when no CUDA device is present.

use super::*;

use crate::handle::BlasHandle;
use crate::types::Transpose;
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// Relative/absolute tolerance for exact-arithmetic f32 kernels.
const EXACT_REL: f32 = 1e-4;
/// Relative/absolute tolerance for exact-arithmetic f32 kernels.
const EXACT_ABS: f32 = 1e-4;
/// Relative/absolute tolerance for SFU-approximation f32 kernels.
const APPROX_REL: f32 = 5e-3;
/// Relative/absolute tolerance for SFU-approximation f32 kernels.
const APPROX_ABS: f32 = 5e-3;
/// Relative/absolute tolerance for exact-arithmetic f64 kernels. `add.f64`,
/// `mul.f64`, `sqrt.rn.f64`, and negation/relu are IEEE-754 correctly rounded
/// on device, so they agree with the f64 CPU oracle to a few ULP — orders of
/// magnitude tighter than the ~1e-7 error an accidental f32 evaluation shows.
const EXACT_REL_F64: f64 = 1e-12;
/// Relative/absolute tolerance for exact-arithmetic f64 kernels.
const EXACT_ABS_F64: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Shared host helpers
// ---------------------------------------------------------------------------

/// Builds a deterministic `f32` vector uniform in `[lo, hi)`.
fn rand_f32(rng: &mut Lcg, len: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..len).map(|_| rng.range_f32(lo, hi)).collect()
}

/// Builds a deterministic `f64` vector uniform in `[lo, hi)`.
fn rand_f64(rng: &mut Lcg, len: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..len).map(|_| rng.range_f64(lo, hi)).collect()
}

/// Launches a unary `(input, output)` production op and returns the device
/// result copied back to the host.
fn run_unary_f32<F>(fx: &GpuFixture, op: F, input: &[f32]) -> Vec<f32>
where
    F: Fn(
        &BlasHandle,
        u32,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> crate::error::BlasResult<()>,
{
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let x = DeviceBuffer::from_host(input).expect("x upload");
    let mut out = DeviceBuffer::from_host(&vec![0.0f32; input.len()]).expect("out alloc");
    op(&handle, input.len() as u32, &x, &mut out).expect("unary launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; input.len()];
    out.copy_to_host(&mut got).expect("d2h");
    got
}

/// Launches a binary `(a, b, c)` production op and returns C copied back.
fn run_binary_f32<F>(fx: &GpuFixture, op: F, a: &[f32], b: &[f32]) -> Vec<f32>
where
    F: Fn(
        &BlasHandle,
        u32,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> crate::error::BlasResult<()>,
{
    assert_eq!(a.len(), b.len(), "binary inputs length mismatch");
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(a).expect("a upload");
    let d_b = DeviceBuffer::from_host(b).expect("b upload");
    let mut d_c = DeviceBuffer::from_host(&vec![0.0f32; a.len()]).expect("c alloc");
    op(&handle, a.len() as u32, &d_a, &d_b, &mut d_c).expect("binary launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; a.len()];
    d_c.copy_to_host(&mut got).expect("d2h");
    got
}

/// Launches a unary `(input, output)` f64 production op and returns the device
/// result copied back to the host.
fn run_unary_f64<F>(fx: &GpuFixture, op: F, input: &[f64]) -> Vec<f64>
where
    F: Fn(
        &BlasHandle,
        u32,
        &DeviceBuffer<f64>,
        &mut DeviceBuffer<f64>,
    ) -> crate::error::BlasResult<()>,
{
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let x = DeviceBuffer::from_host(input).expect("x upload");
    let mut out = DeviceBuffer::from_host(&vec![0.0f64; input.len()]).expect("out alloc");
    op(&handle, input.len() as u32, &x, &mut out).expect("unary launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f64; input.len()];
    out.copy_to_host(&mut got).expect("d2h");
    got
}

/// Launches a binary `(a, b, c)` f64 production op and returns C copied back.
fn run_binary_f64<F>(fx: &GpuFixture, op: F, a: &[f64], b: &[f64]) -> Vec<f64>
where
    F: Fn(
        &BlasHandle,
        u32,
        &DeviceBuffer<f64>,
        &DeviceBuffer<f64>,
        &mut DeviceBuffer<f64>,
    ) -> crate::error::BlasResult<()>,
{
    assert_eq!(a.len(), b.len(), "binary inputs length mismatch");
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(a).expect("a upload");
    let d_b = DeviceBuffer::from_host(b).expect("b upload");
    let mut d_c = DeviceBuffer::from_host(&vec![0.0f64; a.len()]).expect("c alloc");
    op(&handle, a.len() as u32, &d_a, &d_b, &mut d_c).expect("binary launch");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f64; a.len()];
    d_c.copy_to_host(&mut got).expect("d2h");
    got
}

/// CPU sigmoid (matches the kernel's mathematical intent: `1/(1+e^-x)`).
fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Default multi-block element count: not a multiple of the 256 block size, so
/// the in-kernel `tid < n` bounds guard is exercised (a missing guard would
/// fault when the trailing partial block writes past the n-sized output).
const N_MULTIBLOCK: usize = 1031;

// ===========================================================================
// 1. Unary elementwise — activations & math (f32)
// ===========================================================================

#[test]
fn unary_relu_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0001);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    let got = run_unary_f32(&fx, crate::elementwise::relu::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "relu");
}

#[test]
fn unary_neg_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0002);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -5.0, 5.0);
    let got = run_unary_f32(&fx, crate::elementwise::neg::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| -v).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "neg");
}

#[test]
fn unary_abs_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0003);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -5.0, 5.0);
    let got = run_unary_f32(&fx, crate::elementwise::abs_val::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.abs()).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "abs");
}

#[test]
fn unary_sqrt_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0004);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, 0.01, 16.0);
    let got = run_unary_f32(&fx, crate::elementwise::sqrt::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.sqrt()).collect();
    // sqrt.rn is correctly rounded — exact within f32 ulp.
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "sqrt");
}

#[test]
fn unary_rsqrt_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0005);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, 0.1, 16.0);
    let got = run_unary_f32(&fx, crate::elementwise::rsqrt::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| 1.0 / v.sqrt()).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "rsqrt");
}

#[test]
fn unary_exp_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0006);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    let got = run_unary_f32(&fx, crate::elementwise::exp::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.exp()).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "exp");
}

#[test]
fn unary_log_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0007);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, 0.05, 8.0);
    let got = run_unary_f32(&fx, crate::elementwise::log::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.ln()).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "log");
}

#[test]
fn unary_sigmoid_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0008);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -6.0, 6.0);
    let got = run_unary_f32(&fx, crate::elementwise::sigmoid::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| cpu_sigmoid(v)).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "sigmoid");
}

#[test]
fn unary_silu_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0009);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -6.0, 6.0);
    let got = run_unary_f32(&fx, crate::elementwise::silu::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v * cpu_sigmoid(v)).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "silu");
}

#[test]
fn unary_tanh_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000A);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_unary_f32(&fx, crate::elementwise::tanh_activation::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.tanh()).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "tanh");
}

#[test]
fn unary_gelu_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000B);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_unary_f32(&fx, crate::elementwise::gelu::<f32>, &x);
    // GELU tanh approximation, matching the kernel formula exactly.
    let exp: Vec<f32> = x
        .iter()
        .map(|&v| {
            let k0: f32 = 0.797_884_6;
            let k1: f32 = 0.044_715;
            let inner = k0 * k1.mul_add(v * v * v, v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "gelu");
}

#[test]
fn unary_softplus_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000C);
    // Keep |x| modest: softplus = ln(1+e^x), and large x amplifies the SFU exp
    // error through the outer log.
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_unary_f32(&fx, crate::elementwise::softplus::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.exp().ln_1p()).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "softplus");
}

#[test]
fn unary_hard_sigmoid_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000D);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -8.0, 8.0);
    let got = run_unary_f32(&fx, crate::elementwise::hard_sigmoid::<f32>, &x);
    // ONNX hard-sigmoid: max(0, min(1, 0.2*x + 0.5)) == clamp(0.2*x+0.5, 0, 1).
    let exp: Vec<f32> = x
        .iter()
        .map(|&v| (0.2f32 * v + 0.5).clamp(0.0, 1.0))
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "hard_sigmoid");
}

#[test]
fn unary_hard_swish_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000E);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -8.0, 8.0);
    let got = run_unary_f32(&fx, crate::elementwise::hard_swish::<f32>, &x);
    // x * max(0, min(6, x+3)) / 6 == x * clamp(x+3, 0, 6) / 6.
    let exp: Vec<f32> = x
        .iter()
        .map(|&v| v * (v + 3.0).clamp(0.0, 6.0) / 6.0)
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "hard_swish");
}

#[test]
fn unary_leaky_relu_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_000F);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -5.0, 5.0);
    let got = run_unary_f32(&fx, crate::elementwise::leaky_relu::<f32>, &x);
    let exp: Vec<f32> = x
        .iter()
        .map(|&v| if v >= 0.0 { v } else { 0.01 * v })
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "leaky_relu");
}

#[test]
fn unary_ceil_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0010);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -7.5, 7.5);
    let got = run_unary_f32(&fx, crate::elementwise::ceil::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.ceil()).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "ceil");
}

#[test]
fn unary_floor_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0011);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -7.5, 7.5);
    let got = run_unary_f32(&fx, crate::elementwise::floor::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| v.floor()).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "floor");
}

#[test]
fn unary_one_minus_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0012);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    let got = run_unary_f32(&fx, crate::elementwise::one_minus::<f32>, &x);
    let exp: Vec<f32> = x.iter().map(|&v| 1.0 - v).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "one_minus");
}

#[test]
fn unary_scale_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5151_0013);
    let x = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let alpha = -1.75f32;

    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_x = DeviceBuffer::from_host(&x).expect("x upload");
    let mut d_out = DeviceBuffer::from_host(&vec![0.0f32; x.len()]).expect("out alloc");
    crate::elementwise::scale(&handle, x.len() as u32, alpha, &d_x, &mut d_out).expect("scale");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; x.len()];
    d_out.copy_to_host(&mut got).expect("d2h");

    let exp: Vec<f32> = x.iter().map(|&v| alpha * v).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "scale");
}

// ===========================================================================
// 2. Unary/binary elementwise — f64 exact-arithmetic path
// ===========================================================================
//
// The f64 elementwise path was historically broken: `oxicuda-ptx`'s `raw_ptx`
// register inference declared every `%f_*` register `.f32`, so an f64 kernel
// (which issues `ld.global.f64` / `add.f64` / ...) declared its registers at the
// wrong width and ptxas rejected the whole module at load. That is now fixed —
// the elementwise template emits `%fd_*` names for f64 and `raw_ptx` declares
// those `.b64` — so the *exact-arithmetic* f64 kernels JIT, load, and run on
// device. These tests validate `relu`/`neg`/`sqrt`/`add`/`mul` against an
// independent f64 CPU oracle at a few-ULP tolerance; an accidental f32
// evaluation would miss by ~1e-7 and fail loudly (see `EXACT_REL_F64`).
//
// SFU-approximation ops (sigmoid/tanh/exp/log/rsqrt/silu/gelu/softplus/pow)
// stay f32-only — the `ex2.approx` / `lg2.approx` / `rsqrt.approx` units have no
// `.f64` form — and the template rejects them up front at f64 with an `f32-only`
// error (e.g. `ptx_template_rejects_sigmoid_f64` in `elementwise/unary.rs`).

#[test]
fn unary_relu_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x6464_0001);
    let x = rand_f64(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    let got = run_unary_f64(&fx, crate::elementwise::relu::<f64>, &x);
    let exp: Vec<f64> = x.iter().map(|&v| v.max(0.0)).collect();
    assert_close_f64(&got, &exp, EXACT_REL_F64, EXACT_ABS_F64, "relu_f64");
}

#[test]
fn unary_neg_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x6464_0002);
    let x = rand_f64(&mut rng, N_MULTIBLOCK, -5.0, 5.0);
    let got = run_unary_f64(&fx, crate::elementwise::neg::<f64>, &x);
    let exp: Vec<f64> = x.iter().map(|&v| -v).collect();
    assert_close_f64(&got, &exp, EXACT_REL_F64, EXACT_ABS_F64, "neg_f64");
}

#[test]
fn unary_sqrt_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x6464_0003);
    let x = rand_f64(&mut rng, N_MULTIBLOCK, 0.01, 16.0);
    let got = run_unary_f64(&fx, crate::elementwise::sqrt::<f64>, &x);
    // sqrt.rn.f64 is correctly rounded — matches the f64 oracle to <1 ULP.
    let exp: Vec<f64> = x.iter().map(|&v| v.sqrt()).collect();
    assert_close_f64(&got, &exp, EXACT_REL_F64, EXACT_ABS_F64, "sqrt_f64");
}

#[test]
fn binary_add_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x6464_0004);
    let a = rand_f64(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f64(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f64(&fx, crate::elementwise::add::<f64>, &a, &b);
    let exp: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    assert_close_f64(&got, &exp, EXACT_REL_F64, EXACT_ABS_F64, "add_f64");
}

#[test]
fn binary_mul_f64_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x6464_0005);
    let a = rand_f64(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f64(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f64(&fx, crate::elementwise::mul::<f64>, &a, &b);
    let exp: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    assert_close_f64(&got, &exp, EXACT_REL_F64, EXACT_ABS_F64, "mul_f64");
}

// ===========================================================================
// 3. Binary elementwise — arithmetic (f32)
// ===========================================================================

#[test]
fn binary_add_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0001);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::add::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "add");
}

#[test]
fn binary_sub_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0002);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::sub::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x - y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "sub");
}

#[test]
fn binary_mul_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0003);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::mul::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "mul");
}

#[test]
fn binary_div_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0004);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    // Keep |b| away from zero so div.rn is well-conditioned.
    let b: Vec<f32> = rand_f32(&mut rng, N_MULTIBLOCK, 0.5, 3.0)
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
        .collect();
    let got = run_binary_f32(&fx, crate::elementwise::div::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x / y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "div");
}

#[test]
fn binary_pow_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0005);
    // lg2.approx requires a positive base; keep base/exponent modest.
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.2, 3.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.5, 2.5);
    let got = run_binary_f32(&fx, crate::elementwise::pow::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.powf(y)).collect();
    assert_close_f32(&got, &exp, APPROX_REL, APPROX_ABS, "pow");
}

#[test]
fn binary_min_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0006);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::min::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "min");
}

#[test]
fn binary_max_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0007);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::max::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "max");
}

#[test]
fn binary_or_max_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7373_0008);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let got = run_binary_f32(&fx, crate::elementwise::or_max::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "or_max");
}

// ===========================================================================
// 4. Binary elementwise — fuzzy logic (f32)
// ===========================================================================

#[test]
fn binary_or_prob_sum_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x8484_0001);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let got = run_binary_f32(&fx, crate::elementwise::or_prob_sum::<f32>, &a, &b);
    // Kernel op order: t = a*b; s = a - t; c = s + b.
    let exp: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| {
            let t = x * y;
            (x - t) + y
        })
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "or_prob_sum");
}

#[test]
fn binary_nand_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x8484_0002);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let got = run_binary_f32(&fx, crate::elementwise::nand::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| 1.0 - x * y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "nand");
}

#[test]
fn binary_nor_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x8484_0003);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let got = run_binary_f32(&fx, crate::elementwise::nor::<f32>, &a, &b);
    // Kernel op order: t = a*b; s = a - t; u = s + b; c = 1 - u.
    let exp: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| {
            let t = x * y;
            let u = (x - t) + y;
            1.0 - u
        })
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "nor");
}

#[test]
fn binary_xor_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x8484_0004);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, 0.0, 1.0);
    let got = run_binary_f32(&fx, crate::elementwise::xor::<f32>, &a, &b);
    // Kernel op order: s = a + b; t = a*b; t2 = 2*t; c = s - t2.
    let exp: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| {
            let s = x + y;
            let t2 = 2.0 * (x * y);
            s - t2
        })
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "xor");
}

// ===========================================================================
// 5. Binary elementwise — comparisons (f32)
// ===========================================================================

/// Drives a comparison op and asserts the device 1.0/0.0 mask equals `pred`.
fn check_cmp<F, P>(seed: u64, op: F, pred: P, tag: &str)
where
    F: Fn(
        &BlasHandle,
        u32,
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> crate::error::BlasResult<()>,
    P: Fn(f32, f32) -> bool,
{
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(seed);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    // Make ~half the elements exactly equal so eq/ne/le/ge see both branches.
    let mut b = rand_f32(&mut rng, N_MULTIBLOCK, -3.0, 3.0);
    for i in (0..b.len()).step_by(2) {
        b[i] = a[i];
    }
    let got = run_binary_f32(&fx, op, &a, &b);
    let exp: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| if pred(x, y) { 1.0 } else { 0.0 })
        .collect();
    assert_close_f32(&got, &exp, 1e-6, 1e-6, tag);
}

#[test]
#[allow(clippy::float_cmp)]
fn binary_cmp_eq_f32_matches_host() {
    check_cmp(
        0x9595_0001,
        crate::elementwise::cmp_eq::<f32>,
        |x, y| x == y,
        "cmp_eq",
    );
}

#[test]
#[allow(clippy::float_cmp)]
fn binary_cmp_ne_f32_matches_host() {
    check_cmp(
        0x9595_0002,
        crate::elementwise::cmp_ne::<f32>,
        |x, y| x != y,
        "cmp_ne",
    );
}

#[test]
fn binary_cmp_lt_f32_matches_host() {
    check_cmp(
        0x9595_0003,
        crate::elementwise::cmp_lt::<f32>,
        |x, y| x < y,
        "cmp_lt",
    );
}

#[test]
fn binary_cmp_gt_f32_matches_host() {
    check_cmp(
        0x9595_0004,
        crate::elementwise::cmp_gt::<f32>,
        |x, y| x > y,
        "cmp_gt",
    );
}

#[test]
fn binary_cmp_le_f32_matches_host() {
    check_cmp(
        0x9595_0005,
        crate::elementwise::cmp_le::<f32>,
        |x, y| x <= y,
        "cmp_le",
    );
}

#[test]
fn binary_cmp_ge_f32_matches_host() {
    check_cmp(
        0x9595_0006,
        crate::elementwise::cmp_ge::<f32>,
        |x, y| x >= y,
        "cmp_ge",
    );
}

// ===========================================================================
// 6. Binary elementwise — fused ops & f64 (f32 + f64)
// ===========================================================================

#[test]
fn binary_fused_add_relu_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xA6A6_0001);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let got = run_binary_f32(&fx, crate::elementwise::fused_add_relu::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| (x + y).max(0.0)).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "fused_add_relu");
}

/// Drives `fused_scale_add(alpha, beta)` and returns the device result.
fn run_fused_scale_add(fx: &GpuFixture, alpha: f32, beta: f32, a: &[f32], b: &[f32]) -> Vec<f32> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(a).expect("a upload");
    let d_b = DeviceBuffer::from_host(b).expect("b upload");
    let mut d_c = DeviceBuffer::from_host(&vec![0.0f32; a.len()]).expect("c alloc");
    crate::elementwise::fused_scale_add(&handle, a.len() as u32, alpha, &d_a, beta, &d_b, &mut d_c)
        .expect("fused_scale_add");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; a.len()];
    d_c.copy_to_host(&mut got).expect("d2h");
    got
}

#[test]
fn binary_fused_scale_add_f32_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xA6A6_0002);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let (alpha, beta) = (0.75f32, -1.25f32);
    let got = run_fused_scale_add(&fx, alpha, beta, &a, &b);
    let exp: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| alpha * x + beta * y)
        .collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "fused_scale_add");
}

/// `beta = 0` must drop B entirely (the SpMM/CSR5-class bug: a kernel that
/// silently keeps `beta = 1` would re-add B here and mismatch the oracle).
#[test]
fn binary_fused_scale_add_beta_zero_drops_b() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xA6A6_0003);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let alpha = 1.5f32;
    let got = run_fused_scale_add(&fx, alpha, 0.0, &a, &b);
    let exp: Vec<f32> = a.iter().map(|&x| alpha * x).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "fused_scale_add_beta0");
}

/// Symmetric check: `alpha = 0` must drop A entirely.
#[test]
fn binary_fused_scale_add_alpha_zero_drops_a() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xA6A6_0004);
    let a = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let b = rand_f32(&mut rng, N_MULTIBLOCK, -4.0, 4.0);
    let beta = -0.5f32;
    let got = run_fused_scale_add(&fx, 0.0, beta, &a, &b);
    let exp: Vec<f32> = b.iter().map(|&y| beta * y).collect();
    assert_close_f32(&got, &exp, EXACT_REL, EXACT_ABS, "fused_scale_add_alpha0");
}

// ===========================================================================
// 7. Non-vacuous probe — the launch genuinely reads device memory
// ===========================================================================

/// Perturbing one input element must change the corresponding output element,
/// proving the kernel actually reads device memory (not a no-op write).
#[test]
fn binary_add_corruption_probe_is_detected() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xC8C8_0001);
    let a = rand_f32(&mut rng, 512, -2.0, 2.0);
    let b = rand_f32(&mut rng, 512, -2.0, 2.0);

    let clean = run_binary_f32(&fx, crate::elementwise::add::<f32>, &a, &b);
    let exp: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    assert_close_f32(&clean, &exp, EXACT_REL, EXACT_ABS, "probe_clean");

    let mut a_bad = a.clone();
    a_bad[100] += 9.0;
    let dirty = run_binary_f32(&fx, crate::elementwise::add::<f32>, &a_bad, &b);
    assert!(
        (clean[100] - dirty[100]).abs() > 1.0,
        "corrupting input A[100] did not change output[100] — kernel may not read device memory"
    );
    // Untouched elements must stay put.
    assert!(
        (clean[0] - dirty[0]).abs() <= EXACT_ABS,
        "unrelated output element changed unexpectedly"
    );
}

// ===========================================================================
// 8. Complex GEMM / GEMV
// ===========================================================================

/// Geometry of a complex GEMM (transpose modes, logical dims, leading dims).
/// Mirrors the production kernel's index math so the oracle is exact.
#[derive(Clone, Copy)]
struct CGemmGeo {
    transa: Transpose,
    transb: Transpose,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
}

/// CPU oracle for the complex GEMM kernel, computed in f64 from interleaved
/// `(re, im)` host data. Uses the *exact* index formulas the PTX kernel uses:
/// `A[NoTrans] = a[(row*lda + p)*2]`, `A[Trans] = a[(p*lda + row)*2]`, with the
/// imaginary lane negated under `ConjTrans`; `B` symmetric in `col`; the
/// epilogue applies complex `alpha`/`beta` and writes only the `m x n` block.
fn cpu_cgemm(
    geo: &CGemmGeo,
    alpha: (f64, f64),
    beta: (f64, f64),
    a: &[f64],
    b: &[f64],
    c0: &[f64],
) -> Vec<f64> {
    let mut c = c0.to_vec();
    let conj_a = matches!(geo.transa, Transpose::ConjTrans);
    let conj_b = matches!(geo.transb, Transpose::ConjTrans);
    let trans_a = !matches!(geo.transa, Transpose::NoTrans);
    let trans_b = !matches!(geo.transb, Transpose::NoTrans);

    for row in 0..geo.m {
        for col in 0..geo.n {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for p in 0..geo.k {
                let a_idx = if trans_a {
                    p * geo.lda + row
                } else {
                    row * geo.lda + p
                };
                let a_re = a[2 * a_idx];
                let mut a_im = a[2 * a_idx + 1];
                if conj_a {
                    a_im = -a_im;
                }
                let b_idx = if trans_b {
                    col * geo.ldb + p
                } else {
                    p * geo.ldb + col
                };
                let b_re = b[2 * b_idx];
                let mut b_im = b[2 * b_idx + 1];
                if conj_b {
                    b_im = -b_im;
                }
                acc_re += a_re * b_re - a_im * b_im;
                acc_im += a_re * b_im + a_im * b_re;
            }
            let c_idx = row * geo.ldc + col;
            let c_re = c[2 * c_idx];
            let c_im = c[2 * c_idx + 1];
            let res_re = alpha.0 * acc_re - alpha.1 * acc_im + beta.0 * c_re - beta.1 * c_im;
            let res_im = alpha.0 * acc_im + alpha.1 * acc_re + beta.0 * c_im + beta.1 * c_re;
            c[2 * c_idx] = res_re;
            c[2 * c_idx + 1] = res_im;
        }
    }
    c
}

/// Drives the production `complex_gemm::<f32>` and returns C copied back.
fn run_cgemm_f32(
    fx: &GpuFixture,
    geo: &CGemmGeo,
    alpha: (f32, f32),
    beta: (f32, f32),
    a: &[f32],
    b: &[f32],
    c0: &[f32],
) -> Vec<f32> {
    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(a).expect("a upload");
    let d_b = DeviceBuffer::from_host(b).expect("b upload");
    let mut d_c = DeviceBuffer::from_host(c0).expect("c upload");
    crate::complex_gemm::complex_gemm::<f32>(
        &handle, geo.transa, geo.transb, geo.m, geo.n, geo.k, alpha.0, alpha.1, &d_a, geo.lda,
        &d_b, geo.ldb, beta.0, beta.1, &mut d_c, geo.ldc,
    )
    .expect("complex_gemm f32");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; c0.len()];
    d_c.copy_to_host(&mut got).expect("d2h");
    got
}

/// Asserts a square complex GEMM (`dim x dim x dim`, row-major, all lds = dim)
/// matches the oracle for the given transpose modes and complex scalars.
fn assert_square_cgemm_f32(
    fx: &GpuFixture,
    seed: u64,
    transa: Transpose,
    transb: Transpose,
    alpha: (f32, f32),
    beta: (f32, f32),
    tag: &str,
) {
    let dim = 24usize; // 24x24 -> 2x2 grid of 16x16 blocks (multi-block path).
    let geo = CGemmGeo {
        transa,
        transb,
        m: dim,
        n: dim,
        k: dim,
        lda: dim,
        ldb: dim,
        ldc: dim,
    };
    let real_len = 2 * dim * dim;
    let mut rng = Lcg::new(seed);
    let a = rand_f32(&mut rng, real_len, -1.0, 1.0);
    let b = rand_f32(&mut rng, real_len, -1.0, 1.0);
    let c0 = rand_f32(&mut rng, real_len, -0.5, 0.5);

    let got = run_cgemm_f32(fx, &geo, alpha, beta, &a, &b, &c0);

    let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let c64: Vec<f64> = c0.iter().map(|&v| v as f64).collect();
    let exp64 = cpu_cgemm(
        &geo,
        (alpha.0 as f64, alpha.1 as f64),
        (beta.0 as f64, beta.1 as f64),
        &a64,
        &b64,
        &c64,
    );
    let exp: Vec<f32> = exp64.iter().map(|&v| v as f32).collect();
    assert_close_f32(&got, &exp, 1e-4, 1e-4, tag);
}

#[test]
fn cgemm_f32_nn_alpha_beta() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemm_f32(
        &fx,
        0xD901_0001,
        Transpose::NoTrans,
        Transpose::NoTrans,
        (0.75, -0.5),
        (1.25, 0.25),
        "cgemm_nn_ab",
    );
}

#[test]
fn cgemm_f32_nn_beta_zero() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // beta = 0 must drop the incoming C (dropped-beta probe).
    assert_square_cgemm_f32(
        &fx,
        0xD901_0002,
        Transpose::NoTrans,
        Transpose::NoTrans,
        (1.0, 0.0),
        (0.0, 0.0),
        "cgemm_nn_beta0",
    );
}

#[test]
fn cgemm_f32_tn() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemm_f32(
        &fx,
        0xD901_0003,
        Transpose::Trans,
        Transpose::NoTrans,
        (0.9, 0.3),
        (-0.4, 0.7),
        "cgemm_tn",
    );
}

#[test]
fn cgemm_f32_nt() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemm_f32(
        &fx,
        0xD901_0004,
        Transpose::NoTrans,
        Transpose::Trans,
        (1.1, -0.2),
        (0.5, -0.5),
        "cgemm_nt",
    );
}

#[test]
fn cgemm_f32_tt() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemm_f32(
        &fx,
        0xD901_0005,
        Transpose::Trans,
        Transpose::Trans,
        (0.6, 0.6),
        (0.8, -0.1),
        "cgemm_tt",
    );
}

#[test]
fn cgemm_f32_conj_cn() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // ConjTrans on A must negate A's imaginary lane.
    assert_square_cgemm_f32(
        &fx,
        0xD901_0006,
        Transpose::ConjTrans,
        Transpose::NoTrans,
        (1.0, 0.0),
        (0.0, 0.0),
        "cgemm_cn",
    );
}

#[test]
fn cgemm_f32_conj_cc() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemm_f32(
        &fx,
        0xD901_0007,
        Transpose::ConjTrans,
        Transpose::ConjTrans,
        (0.7, -0.3),
        (0.2, 0.2),
        "cgemm_cc",
    );
}

#[test]
fn cgemm_f32_nonsquare_true_matmul() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // m <= k <= n with row-major leading dims => distinct, in-bounds elements
    // (a genuine non-square complex matmul, not an aliased view).
    let (m, k, n) = (4usize, 6usize, 8usize);
    let geo = CGemmGeo {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m,
        n,
        k,
        lda: k,
        ldb: n,
        ldc: n,
    };
    let mut rng = Lcg::new(0xD901_0008);
    let a = rand_f32(&mut rng, 2 * geo.lda * k, -1.0, 1.0);
    let b = rand_f32(&mut rng, 2 * geo.ldb * n, -1.0, 1.0);
    let c0 = rand_f32(&mut rng, 2 * geo.ldc * n, -0.5, 0.5);
    let alpha = (0.85f32, -0.35f32);
    let beta = (0.4f32, 0.6f32);

    let got = run_cgemm_f32(&fx, &geo, alpha, beta, &a, &b, &c0);

    let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    let c64: Vec<f64> = c0.iter().map(|&v| v as f64).collect();
    let exp64 = cpu_cgemm(
        &geo,
        (alpha.0 as f64, alpha.1 as f64),
        (beta.0 as f64, beta.1 as f64),
        &a64,
        &b64,
        &c64,
    );
    let exp: Vec<f32> = exp64.iter().map(|&v| v as f32).collect();
    assert_close_f32(&got, &exp, 1e-4, 1e-4, "cgemm_nonsquare");
}

#[test]
fn cgemm_f64_nn_square() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let dim = 16usize;
    let geo = CGemmGeo {
        transa: Transpose::NoTrans,
        transb: Transpose::NoTrans,
        m: dim,
        n: dim,
        k: dim,
        lda: dim,
        ldb: dim,
        ldc: dim,
    };
    let real_len = 2 * dim * dim;
    let mut rng = Lcg::new(0xDA02_0001);
    let a = rand_f64(&mut rng, real_len, -1.0, 1.0);
    let b = rand_f64(&mut rng, real_len, -1.0, 1.0);
    let c0 = rand_f64(&mut rng, real_len, -0.5, 0.5);
    let alpha = (0.75f64, -0.5f64);
    let beta = (1.25f64, 0.25f64);

    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&a).expect("a");
    let d_b = DeviceBuffer::from_host(&b).expect("b");
    let mut d_c = DeviceBuffer::from_host(&c0).expect("c");
    crate::complex_gemm::complex_gemm::<f64>(
        &handle, geo.transa, geo.transb, geo.m, geo.n, geo.k, alpha.0, alpha.1, &d_a, geo.lda,
        &d_b, geo.ldb, beta.0, beta.1, &mut d_c, geo.ldc,
    )
    .expect("complex_gemm f64");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f64; real_len];
    d_c.copy_to_host(&mut got).expect("d2h");

    let exp = cpu_cgemm(&geo, alpha, beta, &a, &b, &c0);
    assert_close_f64(&got, &exp, 1e-10, 1e-10, "cgemm_f64_nn");
}

// --- Complex GEMV ----------------------------------------------------------

/// Geometry of a complex GEMV (matrix is `m x n` complex; vectors strided).
#[derive(Clone, Copy)]
struct CGemvGeo {
    trans: Transpose,
    m: usize,
    n: usize,
    lda: usize,
    incx: usize,
    incy: usize,
}

/// CPU oracle for the complex GEMV kernel, computed in f64. Mirrors the PTX:
/// `output_len`/`inner_len` swap under transpose, `A` indexed as
/// `(gid*lda + p)` (NoTrans) or `(p*lda + gid)` (Trans/Conj) with the
/// imaginary lane negated under `ConjTrans`; `x`/`y` use complex-unit strides.
fn cpu_cgemv(
    geo: &CGemvGeo,
    alpha: (f64, f64),
    beta: (f64, f64),
    a: &[f64],
    x: &[f64],
    y0: &[f64],
) -> Vec<f64> {
    let mut y = y0.to_vec();
    let trans = !matches!(geo.trans, Transpose::NoTrans);
    let conj = matches!(geo.trans, Transpose::ConjTrans);
    let (output_len, inner_len) = if trans {
        (geo.n, geo.m)
    } else {
        (geo.m, geo.n)
    };

    for gid in 0..output_len {
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        for p in 0..inner_len {
            let a_idx = if trans {
                p * geo.lda + gid
            } else {
                gid * geo.lda + p
            };
            let a_re = a[2 * a_idx];
            let mut a_im = a[2 * a_idx + 1];
            if conj {
                a_im = -a_im;
            }
            let x_off = p * geo.incx;
            let x_re = x[2 * x_off];
            let x_im = x[2 * x_off + 1];
            acc_re += a_re * x_re - a_im * x_im;
            acc_im += a_re * x_im + a_im * x_re;
        }
        let y_off = gid * geo.incy;
        let y_re = y[2 * y_off];
        let y_im = y[2 * y_off + 1];
        let res_re = alpha.0 * acc_re - alpha.1 * acc_im + beta.0 * y_re - beta.1 * y_im;
        let res_im = alpha.0 * acc_im + alpha.1 * acc_re + beta.0 * y_im + beta.1 * y_re;
        y[2 * y_off] = res_re;
        y[2 * y_off + 1] = res_im;
    }
    y
}

/// Drives the production `complex_gemv::<f32>` (square A) and checks the oracle.
/// `inc` carries the `(incx, incy)` complex-element strides.
fn assert_square_cgemv_f32(
    fx: &GpuFixture,
    seed: u64,
    trans: Transpose,
    inc: (usize, usize),
    alpha: (f32, f32),
    beta: (f32, f32),
    tag: &str,
) {
    let (incx, incy) = inc;
    let dim = 20usize;
    let geo = CGemvGeo {
        trans,
        m: dim,
        n: dim,
        lda: dim,
        incx,
        incy,
    };
    let mut rng = Lcg::new(seed);
    // For square A both x and y span `dim` complex elements.
    let a = rand_f32(&mut rng, 2 * dim * dim, -1.0, 1.0);
    let x = rand_f32(&mut rng, 2 * (1 + (dim - 1) * incx), -1.0, 1.0);
    let y0 = rand_f32(&mut rng, 2 * (1 + (dim - 1) * incy), -0.5, 0.5);

    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&a).expect("a");
    let d_x = DeviceBuffer::from_host(&x).expect("x");
    let mut d_y = DeviceBuffer::from_host(&y0).expect("y");
    crate::complex_gemm::complex_gemv::<f32>(
        &handle, geo.trans, geo.m, geo.n, alpha.0, alpha.1, &d_a, geo.lda, &d_x, geo.incx, beta.0,
        beta.1, &mut d_y, geo.incy,
    )
    .expect("complex_gemv f32");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f32; y0.len()];
    d_y.copy_to_host(&mut got).expect("d2h");

    let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y64: Vec<f64> = y0.iter().map(|&v| v as f64).collect();
    let exp64 = cpu_cgemv(
        &geo,
        (alpha.0 as f64, alpha.1 as f64),
        (beta.0 as f64, beta.1 as f64),
        &a64,
        &x64,
        &y64,
    );
    let exp: Vec<f32> = exp64.iter().map(|&v| v as f32).collect();
    assert_close_f32(&got, &exp, 1e-4, 1e-4, tag);
}

#[test]
fn cgemv_f32_notrans_unit_stride() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemv_f32(
        &fx,
        0xE101_0001,
        Transpose::NoTrans,
        (1, 1),
        (0.8, -0.4),
        (0.5, 0.25),
        "cgemv_n",
    );
}

#[test]
fn cgemv_f32_notrans_strided() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Non-unit complex-element strides on both vectors.
    assert_square_cgemv_f32(
        &fx,
        0xE101_0002,
        Transpose::NoTrans,
        (2, 3),
        (1.0, 0.0),
        (0.0, 0.0),
        "cgemv_n_strided",
    );
}

#[test]
fn cgemv_f32_trans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemv_f32(
        &fx,
        0xE101_0003,
        Transpose::Trans,
        (1, 2),
        (0.9, 0.3),
        (-0.3, 0.6),
        "cgemv_t",
    );
}

#[test]
fn cgemv_f32_conj_trans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    assert_square_cgemv_f32(
        &fx,
        0xE101_0004,
        Transpose::ConjTrans,
        (2, 1),
        (0.7, -0.7),
        (0.4, 0.1),
        "cgemv_c",
    );
}

#[test]
fn cgemv_f64_notrans() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let dim = 16usize;
    let (incx, incy) = (1usize, 1usize);
    let geo = CGemvGeo {
        trans: Transpose::NoTrans,
        m: dim,
        n: dim,
        lda: dim,
        incx,
        incy,
    };
    let mut rng = Lcg::new(0xE202_0001);
    let a = rand_f64(&mut rng, 2 * dim * dim, -1.0, 1.0);
    let x = rand_f64(&mut rng, 2 * (1 + (dim - 1) * incx), -1.0, 1.0);
    let y0 = rand_f64(&mut rng, 2 * (1 + (dim - 1) * incy), -0.5, 0.5);
    let alpha = (0.8f64, -0.4f64);
    let beta = (0.5f64, 0.25f64);

    let handle = BlasHandle::new(&fx.ctx).expect("blas handle");
    let d_a = DeviceBuffer::from_host(&a).expect("a");
    let d_x = DeviceBuffer::from_host(&x).expect("x");
    let mut d_y = DeviceBuffer::from_host(&y0).expect("y");
    crate::complex_gemm::complex_gemv::<f64>(
        &handle, geo.trans, geo.m, geo.n, alpha.0, alpha.1, &d_a, geo.lda, &d_x, geo.incx, beta.0,
        beta.1, &mut d_y, geo.incy,
    )
    .expect("complex_gemv f64");
    handle.stream().synchronize().expect("sync");
    let mut got = vec![0.0f64; y0.len()];
    d_y.copy_to_host(&mut got).expect("d2h");

    let exp = cpu_cgemv(&geo, alpha, beta, &a, &x, &y0);
    assert_close_f64(&got, &exp, 1e-10, 1e-10, "cgemv_f64_n");
}
