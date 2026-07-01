//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it through `oxicuda-launch`, copies results back, and asserts
//! numerical equivalence to the crate's CPU reference. The launch ABI mirrors
//! the working `oxicuda-snn` pattern: device buffers as `.param .u64`
//! (CUdeviceptr), scalars as the matching Rust scalar in declared order.
//!
//! ## Oracle strength tiers
//!
//! * **CRATE ORACLE** (strongest) — compared bit-for-bit to a `pub` CPU
//!   function the kernel is meant to mirror:
//!   `xor_bind` (binary_bind), `bundle_majority` (bundle_binary, odd K so
//!   tie-branch is never taken), `cyclic_shift` (cyclic_shift_i32),
//!   `hamming_dist` (hamming_count), `complex_bind` (complex_bind),
//!   `hd_classify` (argmax_cosine_binary).
//!
//! * **INDEPENDENT HOST RE-DERIVATION** — the kernel has no dedicated crate
//!   function returning the same intermediate quantities: `cosine_sim` writes
//!   three partial sums (dot, norm_a², norm_b²) via f32 atomics; the oracle
//!   is an independent f32 sequential accumulation. The derived cosine is
//!   compared within 1e-3 relative to cover non-deterministic GPU atomic
//!   reordering (~3e-5 expected error for n=256) while catching any formula
//!   error by orders of magnitude.
//!
//! Every test returns early when no CUDA device is present, keeping the
//! suite green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, which creates and makes the context
/// current on the calling thread. The `Arc<Context>` must be kept alive for
/// the whole test; nextest runs each test in its own process, so a per-test
/// context is fine.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

/// Acquire a GPU fixture, or `None` when no driver or device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = (major * 10 + minor) as u32;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// JIT-compile `ptx` and look up `entry`, panicking on failure with the real
/// error message so PTX bugs are surfaced rather than silently skipped.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a 1-D grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

/// Relative-with-absolute-floor closeness test for f32 comparisons.
fn close_f32(a: f32, b: f32, rel: f32, abs_tol: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs_tol
}

/// Worst (relative, absolute) divergence over two equal-length f32 slices.
fn worst_diff_f32(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        let a = (g - c).abs();
        if a > worst_abs {
            worst_abs = a;
        }
        let denom = g.abs().max(c.abs());
        if denom > 0.0 {
            let r = a / denom;
            if r > worst_rel {
                worst_rel = r;
            }
        }
    }
    (worst_rel, worst_abs)
}

// ── 1. xor_bind_kernel — CRATE ORACLE (ops::binding::binary_bind) ────────────
//
// For ±1 i8 encoding, XOR-bind is element-wise sign product: out[i] = a[i]*b[i].
// The GPU kernel loads each i8 as s32, multiplies, stores as s8.
// The CPU `binary_bind` computes the same product directly.
// Expected: bit-exact (integer arithmetic with values in {-1, +1}).

#[test]
fn xor_bind_matches_cpu() {
    use crate::ops::binding::binary_bind;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0xDEAD_BEEF_1234_5678);
    let mut a = vec![0i8; n];
    let mut b = vec![0i8; n];
    rng.fill_binary(&mut a);
    rng.fill_binary(&mut b);

    let out_cpu = binary_bind(&a, &b).expect("binary_bind");

    let ptx = crate::ptx_kernels::xor_bind_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "xor_bind_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<i8>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<i8>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<i8>::from_host(&vec![0i8; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch xor_bind_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0i8; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for k in 0..n {
        assert_eq!(
            out_gpu[k], out_cpu[k],
            "xor_bind mismatch at {k}: gpu={} cpu={}",
            out_gpu[k], out_cpu[k]
        );
    }
}

// ── 2. bundle_majority_kernel — CRATE ORACLE (ops::bundling::bundle_binary) ──
//
// Kernel: row-major matrix (K rows × N cols of i8 ±1), one thread per column.
// Accumulates column sums, outputs 1 when sum>0, else -1.
//
// CPU oracle: bundle_binary with rng for tie-breaking.
// DESIGN CHOICE: use odd K=5 so column sums are always odd (±1,±3,±5) and
// can never be zero. This makes the rng tie-break branch unreachable, giving
// a deterministic comparison that is genuinely failable if the kernel gets
// any sum element wrong.

#[test]
fn bundle_majority_matches_cpu() {
    use crate::ops::bundling::bundle_binary;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let kk = 5_usize; // odd K → no ties
    let n = 64_usize;
    let mut rng = LcgRng::new(0xCAFE_BABE_0000_0001);
    let hvs: Vec<Vec<i8>> = (0..kk)
        .map(|_| {
            let mut v = vec![0i8; n];
            rng.fill_binary(&mut v);
            v
        })
        .collect();

    // Precondition: no ties (guaranteed by odd K; assert to catch test-setup bugs)
    for j in 0..n {
        let sum: i32 = hvs.iter().map(|hv| hv[j] as i32).sum();
        assert_ne!(
            sum, 0,
            "test setup error: tie at column {j} (odd K should prevent this)"
        );
    }

    // Flatten row-major [K][N] for the GPU matrix parameter
    let matrix: Vec<i8> = hvs.iter().flat_map(|hv| hv.iter().copied()).collect();

    // CPU oracle — rng never called because no ties
    let mut dummy_rng = LcgRng::new(0);
    let out_cpu = bundle_binary(&hvs, &mut dummy_rng).expect("bundle_binary");

    let ptx = crate::ptx_kernels::bundle_majority_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bundle_majority_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_matrix = DeviceBuffer::<i8>::from_host(&matrix).expect("d_matrix");
    let d_out = DeviceBuffer::<i8>::from_host(&vec![0i8; n]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_matrix.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                kk as u32,
            ),
        )
        .expect("launch bundle_majority_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0i8; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for k in 0..n {
        assert_eq!(
            out_gpu[k], out_cpu[k],
            "bundle_majority mismatch at {k}: gpu={} cpu={}",
            out_gpu[k], out_cpu[k]
        );
    }
}

// ── 3. cyclic_shift_kernel — CRATE ORACLE (ops::permutation::cyclic_shift_i32) ─
//
// Kernel: out[i] = in[(i + k) % n] for i32 elements.
// GPU uses `rem.u32` for the modulo, which matches the host `%` for u32.
// Expected: bit-exact (no floating point, pure integer index arithmetic).

#[test]
fn cyclic_shift_matches_cpu() {
    use crate::ops::permutation::cyclic_shift_i32;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let shift = 37_usize;
    let mut rng = LcgRng::new(0x1357_2468_9BDF_0ACE);
    // Generate arbitrary i32 values via next_u32 cast — covers full bit range
    let hv: Vec<i32> = (0..n).map(|_| rng.next_u32() as i32).collect();

    let out_cpu = cyclic_shift_i32(&hv, shift).expect("cyclic_shift_i32");

    let ptx = crate::ptx_kernels::cyclic_shift_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cyclic_shift_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<i32>::from_host(&hv).expect("d_in");
    let d_out = DeviceBuffer::<i32>::from_host(&vec![0i32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                shift as u32,
            ),
        )
        .expect("launch cyclic_shift_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0i32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for k in 0..n {
        assert_eq!(
            out_gpu[k], out_cpu[k],
            "cyclic_shift mismatch at {k}: gpu={} cpu={}",
            out_gpu[k], out_cpu[k]
        );
    }
}

// ── 4. cosine_sim_kernel — INDEPENDENT HOST RE-DERIVATION ────────────────────
//
// HONEST SCOPE: The kernel accumulates three f32 partial sums (dot product,
// sum-of-squares for a, sum-of-squares for b) via `red.global.add.f32`. There
// is no crate function that returns these three intermediate quantities; the
// crate's `cosine_real` uses f64 accumulation and returns a single f32 cosine.
//
// Oracle: independent f32 sequential accumulation of the same three quantities,
// matching the kernel's arithmetic type. Non-deterministic GPU atomic reordering
// can permute the summation, introducing up to ~n*eps_f32 ≈ 256 × 1.2e-7 ≈ 3e-5
// relative error vs a sequential host sum. Tolerance 1e-3 gives 30× margin while
// catching any formula error by > 3 orders of magnitude.

#[test]
fn cosine_sim_accumulates_correctly() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0xBEEF_CAFE_4321_FEDC);
    let a: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: f32 sequential accumulation (same type as GPU partial sums).
    let mut dot_host = 0.0_f32;
    let mut norm_a_host = 0.0_f32;
    let mut norm_b_host = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot_host += ai * bi;
        norm_a_host += ai * ai;
        norm_b_host += bi * bi;
    }
    let cosine_host = dot_host / (norm_a_host.sqrt() * norm_b_host.sqrt());

    let ptx = crate::ptx_kernels::cosine_sim_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cosine_sim_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    // Output accumulator buffers MUST be pre-initialized to 0.0 before launch.
    let d_dot = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_dot");
    let d_norm_a = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_norm_a");
    let d_norm_b = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_norm_b");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_dot.as_device_ptr(),
                d_norm_a.as_device_ptr(),
                d_norm_b.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch cosine_sim_kernel");
    stream.synchronize().expect("sync");

    let mut dot_gpu = vec![0.0_f32; 1];
    let mut norm_a_gpu = vec![0.0_f32; 1];
    let mut norm_b_gpu = vec![0.0_f32; 1];
    d_dot.copy_to_host(&mut dot_gpu).expect("copy dot");
    d_norm_a.copy_to_host(&mut norm_a_gpu).expect("copy norm_a");
    d_norm_b.copy_to_host(&mut norm_b_gpu).expect("copy norm_b");

    // Structural: norm values must be positive and finite (random vectors are
    // not the zero vector with overwhelming probability).
    assert!(
        norm_a_gpu[0] > 0.0 && norm_a_gpu[0].is_finite(),
        "norm_a = {} not positive-finite",
        norm_a_gpu[0]
    );
    assert!(
        norm_b_gpu[0] > 0.0 && norm_b_gpu[0].is_finite(),
        "norm_b = {} not positive-finite",
        norm_b_gpu[0]
    );
    assert!(dot_gpu[0].is_finite(), "dot not finite: {}", dot_gpu[0]);

    let cosine_gpu = dot_gpu[0] / (norm_a_gpu[0].sqrt() * norm_b_gpu[0].sqrt());
    // Cosine must be in [-1, 1] (finite random vectors, non-degenerate)
    assert!(
        (-1.0_f32 - 1e-4..=1.0_f32 + 1e-4).contains(&cosine_gpu),
        "cosine_gpu = {cosine_gpu} out of [-1,1]"
    );
    assert!(
        close_f32(cosine_gpu, cosine_host, 1e-3, 1e-5),
        "cosine_sim mismatch: gpu={cosine_gpu:.7} host={cosine_host:.7} (|diff|={:.3e})",
        (cosine_gpu - cosine_host).abs()
    );
}

// ── 5. hamming_dist_kernel — CRATE ORACLE (distance::hamming::hamming_count) ─
//
// Kernel: for each i, if a[i]*b[i] == -1 (signs differ), atomic-add 1 to count.
// The output is a single u32 integer count. Integer atomic-add over booleans is
// commutative and exact regardless of order (unlike float atomics), so the
// result is bit-exact vs the sequential host computation.

#[test]
fn hamming_dist_matches_cpu() {
    use crate::distance::hamming::hamming_count;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x7777_8888_ABCD_EF01);
    let mut a = vec![0i8; n];
    let mut b = vec![0i8; n];
    rng.fill_binary(&mut a);
    rng.fill_binary(&mut b);

    let count_cpu = hamming_count(&a, &b).expect("hamming_count");

    let ptx = crate::ptx_kernels::hamming_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hamming_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<i8>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<i8>::from_host(&b).expect("d_b");
    let d_count = DeviceBuffer::<u32>::from_host(&[0u32]).expect("d_count");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_count.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch hamming_dist_kernel");
    stream.synchronize().expect("sync");

    let mut count_gpu = vec![0u32; 1];
    d_count.copy_to_host(&mut count_gpu).expect("copy count");

    assert_eq!(
        count_gpu[0], count_cpu as u32,
        "hamming_dist mismatch: gpu={} cpu={}",
        count_gpu[0], count_cpu
    );
}

// ── 6. complex_bind_kernel — CRATE ORACLE (vector::complex::complex_bind) ────
//
// FHRR element-wise complex multiply. Storage: interleaved [re_0,im_0,...],
// length = 2*dim; each thread handles one complex element.
//
// GPU arithmetic (PTX mul.f32 + sub.f32 / add.f32, all RN-rounded, no FMA):
//   re_out = a_re*b_re - a_im*b_im
//   im_out = a_re*b_im + a_im*b_re
//
// CPU arithmetic (`complex_bind`): identical two-operation sequences with f32 *.
// Since neither path fuses operations into FMA, results are bit-identical.
// Tolerance 1e-6 relative (≈8 ulp for |v|≤1) gives margin for any surprise
// while catching real formula bugs by many orders of magnitude.

#[test]
fn complex_bind_matches_cpu() {
    use crate::vector::complex::complex_bind;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 128_usize;
    let mut rng = LcgRng::new(0x9999_AAAA_BBBB_CCCC);

    // Random unit-circle complex HVs: re = cos(θ), im = sin(θ)
    let a: Vec<f32> = {
        let mut v = vec![0f32; 2 * dim];
        for i in 0..dim {
            let theta = rng.next_f32() * std::f32::consts::TAU;
            v[2 * i] = theta.cos();
            v[2 * i + 1] = theta.sin();
        }
        v
    };
    let b: Vec<f32> = {
        let mut v = vec![0f32; 2 * dim];
        for i in 0..dim {
            let theta = rng.next_f32() * std::f32::consts::TAU;
            v[2 * i] = theta.cos();
            v[2 * i + 1] = theta.sin();
        }
        v
    };

    let out_cpu = complex_bind(&a, &b).expect("complex_bind");

    let ptx = crate::ptx_kernels::complex_bind_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "complex_bind_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0f32; 2 * dim]).expect("d_out");

    // Kernel iterates over `dim` complex elements (1 thread per element)
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(dim as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                dim as u32,
            ),
        )
        .expect("launch complex_bind_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0f32; 2 * dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (worst_rel, worst_abs) = worst_diff_f32(&out_gpu, &out_cpu);
    for k in 0..(2 * dim) {
        assert!(
            close_f32(out_gpu[k], out_cpu[k], 1e-6, 1e-7),
            "complex_bind mismatch at {k} ({}): gpu={} cpu={} \
             (worst rel={worst_rel:e} abs={worst_abs:e})",
            if k % 2 == 0 { "re" } else { "im" },
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ── 7. hd_classify_kernel — CRATE ORACLE (distance::cosine::argmax_cosine_binary)
//
// Single-threaded kernel: only thread 0 executes the argmax loop (all others
// branch to $HC_DONE immediately). For each class it computes the dot product
// of query vs prototype as a running f32 sum of cvt.rn.f32.s32(q[d]*p[d]),
// divides by dim, and picks the argmax class index.
//
// CPU oracle: argmax_cosine_binary(query, prototypes).
// cosine_binary = dot_i64 / dim_f32. For binary ±1 HVs and dim ≤ 64, the
// integer dot fits exactly in f32, so both sides compute the same cosine value
// and the argmax is bit-exact.

#[test]
fn hd_classify_matches_cpu() {
    use crate::distance::cosine::argmax_cosine_binary;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 64_usize;
    let n_classes = 4_usize;
    let mut rng = LcgRng::new(0xC1A5_5119_C0DE_DEAD);

    // Random binary prototypes, flattened row-major [n_classes][dim]
    let protos_flat: Vec<i8> = {
        let mut v = vec![0i8; n_classes * dim];
        rng.fill_binary(&mut v);
        v
    };
    // Also as Vec<Vec<i8>> for the CPU oracle
    let protos_vecs: Vec<Vec<i8>> = protos_flat.chunks(dim).map(|c| c.to_vec()).collect();

    // Random query
    let mut query = vec![0i8; dim];
    rng.fill_binary(&mut query);

    let class_cpu = argmax_cosine_binary(&query, &protos_vecs).expect("argmax_cosine_binary");

    let ptx = crate::ptx_kernels::hd_classify_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hd_classify_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_query = DeviceBuffer::<i8>::from_host(&query).expect("d_query");
    let d_protos = DeviceBuffer::<i8>::from_host(&protos_flat).expect("d_protos");
    let d_out = DeviceBuffer::<u32>::from_host(&[0u32]).expect("d_out");

    // The kernel gates on global thread index == 0; launch with a single thread.
    let params = LaunchParams::new(1u32, 1u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_query.as_device_ptr(),
                d_protos.as_device_ptr(),
                d_out.as_device_ptr(),
                dim as u32,
                n_classes as u32,
            ),
        )
        .expect("launch hd_classify_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0u32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Class index must be in range
    assert!(
        (out_gpu[0] as usize) < n_classes,
        "hd_classify out={} out of range [0, {n_classes})",
        out_gpu[0]
    );
    assert_eq!(
        out_gpu[0], class_cpu as u32,
        "hd_classify mismatch: gpu={} cpu={}",
        out_gpu[0], class_cpu
    );
}
