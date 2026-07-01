//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to an independent CPU re-derivation of the kernel's documented arithmetic.
//! The launch ABI mirrors the proven `oxicuda-snn` / `oxicuda-ot` harnesses:
//! device buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars
//! as the matching Rust scalar (`.param .u32`), in the kernel's declared
//! parameter order.
//!
//! ## Kernels and oracle tiers (honest accounting)
//!
//! All five kernels in this crate are pure integer-addressed gather / scatter /
//! reduce primitives (no `exp`/`log`/softmax), so the "base-2 vs base-e" PTX bug
//! class does not apply here. Each is validated by an **independent host
//! re-derivation** of the exact element-wise arithmetic the kernel documents:
//!
//! * `tp_col_scatter`   — scatter a `[batch × local_cols]` shard into the
//!   `col_offset` column band of a `[batch × total_cols]` buffer.
//! * `tp_row_all_reduce`— element-wise `buf[i] += accum[i]`.
//! * `sp_seq_chunk_copy`— extract (full→chunk) / insert (chunk→full) a
//!   contiguous token slice (both directions exercised).
//! * `ep_token_scatter` — `expert_buf[slot[t], :] = input[t, :]`.
//! * `ep_token_gather`  — `output[t, :] = expert_buf[slot[t], :]`.
//!
//! The host oracle is written independently of the JIT-compiled PTX, so a wrong
//! constant / shift / index / address in the PTX (or a ptxas miscompile) makes
//! the test genuinely fail. The JIT-load step itself (`Module::from_ptx`) is a
//! real check too: if ptxas rejects the PTX the test panics with the compiler
//! diagnostic.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::SmVersion;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, which both creates the context and makes
/// it current on the calling thread; the returned `Arc<Context>` must be kept
/// alive for the whole test (nextest runs each test in its own process, so a
/// per-test context is fine).
struct GpuFixture {
    ctx: Arc<Context>,
    sm: SmVersion,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let Ok(dev) = Device::get(0) else {
        return None;
    };
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = SmVersion((major * 10 + minor) as u32);
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure here means ptxas rejected the PTX — a real bug
/// in `ptx_kernels.rs`, surfaced as a panic carrying the compiler diagnostic.
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

/// Assert two equal-length f32 slices are bit-identical.
///
/// Every kernel here only *moves* f32 values (no arithmetic except the
/// `add.f32` in `tp_row_all_reduce`), so for pure-copy kernels the GPU result
/// must equal the host re-derivation bit-for-bit.
fn assert_bits_eq(gpu: &[f32], host: &[f32], what: &str) {
    assert_eq!(gpu.len(), host.len(), "{what}: length mismatch");
    for (k, (&g, &h)) in gpu.iter().zip(host.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "{what}: element {k} differs: gpu={g} host={h}"
        );
    }
}

// ===========================================================================
// 1. tp_col_scatter — host re-derivation (shard → column band of full buffer)
// ===========================================================================

#[test]
fn tp_col_scatter_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // tp_rank = 1 of a 2-way column split: this shard owns columns [4, 8) of a
    // 10-column output (the trailing two columns belong to no shard here, which
    // is fine — the kernel only writes its own band).
    let batch = 3_usize;
    let local_cols = 4_usize;
    let total_cols = 10_usize;
    let col_offset = 4_usize;
    let n = batch * local_cols;

    // Deterministic shard contents; distinctive sentinel in the full buffer so an
    // off-by-one column / row write would land on a sentinel and fail.
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let dst_init: Vec<f32> = (0..batch * total_cols)
        .map(|i| -1000.0 - i as f32)
        .collect();

    // Host re-derivation of the documented scatter.
    let mut dst_host = dst_init.clone();
    for (gid, &s) in src.iter().enumerate() {
        let row = gid / local_cols;
        let lcol = gid % local_cols;
        let gcol = lcol + col_offset;
        dst_host[row * total_cols + gcol] = s;
    }

    let ptx = crate::ptx_kernels::tp_col_scatter_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tp_col_scatter");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_src = DeviceBuffer::<f32>::from_host(&src).expect("d_src");
    let d_dst = DeviceBuffer::<f32>::from_host(&dst_init).expect("d_dst");

    // Multiple blocks with a small block size, so the grid-stride loop
    // (step = blockDim * gridDim) is genuinely exercised across block boundaries.
    let block = 8_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr(),
                n as u32,
                total_cols as u32,
                local_cols as u32,
                col_offset as u32,
            ),
        )
        .expect("launch tp_col_scatter");
    stream.synchronize().expect("sync");

    let mut dst_gpu = vec![0.0_f32; batch * total_cols];
    d_dst.copy_to_host(&mut dst_gpu).expect("copy dst");

    assert_bits_eq(&dst_gpu, &dst_host, "tp_col_scatter");
}

// ===========================================================================
// 2. tp_row_all_reduce — host re-derivation (buf[i] += accum[i])
// ===========================================================================

#[test]
fn tp_row_all_reduce_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 17_usize; // intentionally not a multiple of the block size

    let buf: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 1.5).collect();
    let accum: Vec<f32> = (0..n).map(|i| 3.0 - (i as f32) * 0.125).collect();

    // Host re-derivation: in[i] + accum[i]. A single rounding per element on both
    // sides (one `add.f32` each), so the result is bit-exact.
    let host: Vec<f32> = buf.iter().zip(&accum).map(|(&b, &a)| b + a).collect();

    let ptx = crate::ptx_kernels::tp_row_all_reduce_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tp_row_all_reduce");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_buf = DeviceBuffer::<f32>::from_host(&buf).expect("d_buf");
    let d_acc = DeviceBuffer::<f32>::from_host(&accum).expect("d_acc");

    let block = 8_u32; // forces multiple blocks → exercises grid-stride coverage
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_buf.as_device_ptr(), d_acc.as_device_ptr(), n as u32),
        )
        .expect("launch tp_row_all_reduce");
    stream.synchronize().expect("sync");

    let mut buf_gpu = vec![0.0_f32; n];
    d_buf.copy_to_host(&mut buf_gpu).expect("copy buf");

    assert_bits_eq(&buf_gpu, &host, "tp_row_all_reduce");
}

// ===========================================================================
// 3. sp_seq_chunk_copy — host re-derivation, EXTRACT (full → chunk, dir = 0)
// ===========================================================================

#[test]
fn sp_seq_chunk_copy_extract_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let total_tokens = 6_usize;
    let hidden_dim = 4_usize;
    let chunk_start = 2_usize;
    let chunk_len = 3_usize;
    let n = chunk_len * hidden_dim;

    let full: Vec<f32> = (0..total_tokens * hidden_dim)
        .map(|i| (i as f32) * 0.5 - 3.0)
        .collect();
    let chunk_init = vec![0.0_f32; n];

    // Host re-derivation of the extract path.
    let mut chunk_host = chunk_init.clone();
    for (gid, slot) in chunk_host.iter_mut().enumerate() {
        let tok = gid / hidden_dim;
        let feat = gid % hidden_dim;
        let full_tok = tok + chunk_start;
        *slot = full[full_tok * hidden_dim + feat];
    }

    let ptx = crate::ptx_kernels::sp_seq_chunk_copy_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sp_seq_chunk_copy");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_full = DeviceBuffer::<f32>::from_host(&full).expect("d_full");
    let d_chunk = DeviceBuffer::<f32>::from_host(&chunk_init).expect("d_chunk");

    let block = 8_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_full.as_device_ptr(),
                d_chunk.as_device_ptr(),
                chunk_start as u32,
                chunk_len as u32,
                hidden_dim as u32,
                0_u32, // direction = extract
            ),
        )
        .expect("launch sp_seq_chunk_copy (extract)");
    stream.synchronize().expect("sync");

    let mut chunk_gpu = vec![0.0_f32; n];
    d_chunk.copy_to_host(&mut chunk_gpu).expect("copy chunk");

    assert_bits_eq(&chunk_gpu, &chunk_host, "sp_seq_chunk_copy/extract");
}

// ===========================================================================
// 4. sp_seq_chunk_copy — host re-derivation, INSERT (chunk → full, dir = 1)
// ===========================================================================

#[test]
fn sp_seq_chunk_copy_insert_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let total_tokens = 6_usize;
    let hidden_dim = 4_usize;
    let chunk_start = 1_usize;
    let chunk_len = 3_usize;
    let n = chunk_len * hidden_dim;

    // Distinctive sentinel in the full buffer so we can verify the insert touches
    // exactly the [chunk_start, chunk_start+chunk_len) token band and nothing else.
    let full_init: Vec<f32> = (0..total_tokens * hidden_dim)
        .map(|i| -500.0 - i as f32)
        .collect();
    let chunk: Vec<f32> = (0..n).map(|i| (i as f32) * 0.75 + 1.0).collect();

    // Host re-derivation of the insert path.
    let mut full_host = full_init.clone();
    for (gid, &c) in chunk.iter().enumerate() {
        let tok = gid / hidden_dim;
        let feat = gid % hidden_dim;
        let full_tok = tok + chunk_start;
        full_host[full_tok * hidden_dim + feat] = c;
    }

    let ptx = crate::ptx_kernels::sp_seq_chunk_copy_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sp_seq_chunk_copy");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_full = DeviceBuffer::<f32>::from_host(&full_init).expect("d_full");
    let d_chunk = DeviceBuffer::<f32>::from_host(&chunk).expect("d_chunk");

    let block = 8_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_full.as_device_ptr(),
                d_chunk.as_device_ptr(),
                chunk_start as u32,
                chunk_len as u32,
                hidden_dim as u32,
                1_u32, // direction = insert
            ),
        )
        .expect("launch sp_seq_chunk_copy (insert)");
    stream.synchronize().expect("sync");

    let mut full_gpu = vec![0.0_f32; total_tokens * hidden_dim];
    d_full.copy_to_host(&mut full_gpu).expect("copy full");

    assert_bits_eq(&full_gpu, &full_host, "sp_seq_chunk_copy/insert");
}

// ===========================================================================
// 5. ep_token_scatter — host re-derivation (expert_buf[slot[t], :] = input[t, :])
// ===========================================================================

#[test]
fn ep_token_scatter_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 5_usize;
    let hidden_dim = 4_usize;

    let input: Vec<f32> = (0..n_tokens * hidden_dim)
        .map(|i| (i as f32) * 0.5 - 4.0)
        .collect();
    // expert_ids are loaded by the kernel but not used in the destination address
    // (slots are precomputed globally); include a non-trivial assignment anyway.
    let expert_ids: Vec<u32> = vec![0, 1, 0, 2, 1];
    // A permutation of destination slots — distinct so the scatter is a bijection.
    let slots: Vec<u32> = vec![2, 0, 4, 1, 3];
    let buf_init = vec![0.0_f32; n_tokens * hidden_dim];

    // Host re-derivation of the documented scatter.
    let mut buf_host = buf_init.clone();
    for tok in 0..n_tokens {
        let slot = slots[tok] as usize;
        for feat in 0..hidden_dim {
            buf_host[slot * hidden_dim + feat] = input[tok * hidden_dim + feat];
        }
    }

    let ptx = crate::ptx_kernels::ep_token_scatter_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ep_token_scatter");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_input = DeviceBuffer::<f32>::from_host(&input).expect("d_input");
    let d_buf = DeviceBuffer::<f32>::from_host(&buf_init).expect("d_buf");
    let d_ids = DeviceBuffer::<u32>::from_host(&expert_ids).expect("d_ids");
    let d_slots = DeviceBuffer::<u32>::from_host(&slots).expect("d_slots");

    let n = n_tokens * hidden_dim;
    let block = 8_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_input.as_device_ptr(),
                d_buf.as_device_ptr(),
                d_ids.as_device_ptr(),
                d_slots.as_device_ptr(),
                n_tokens as u32,
                hidden_dim as u32,
            ),
        )
        .expect("launch ep_token_scatter");
    stream.synchronize().expect("sync");

    let mut buf_gpu = vec![0.0_f32; n_tokens * hidden_dim];
    d_buf.copy_to_host(&mut buf_gpu).expect("copy buf");

    assert_bits_eq(&buf_gpu, &buf_host, "ep_token_scatter");
}

// ===========================================================================
// 6. ep_token_gather — host re-derivation (output[t, :] = expert_buf[slot[t], :])
// ===========================================================================

#[test]
fn ep_token_gather_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 5_usize;
    let hidden_dim = 4_usize;

    let expert_buf: Vec<f32> = (0..n_tokens * hidden_dim)
        .map(|i| 4.0 - (i as f32) * 0.5)
        .collect();
    let slots: Vec<u32> = vec![2, 0, 4, 1, 3];
    let out_init = vec![0.0_f32; n_tokens * hidden_dim];

    // Host re-derivation: gather is the inverse of scatter.
    let mut out_host = out_init.clone();
    for tok in 0..n_tokens {
        let slot = slots[tok] as usize;
        for feat in 0..hidden_dim {
            out_host[tok * hidden_dim + feat] = expert_buf[slot * hidden_dim + feat];
        }
    }

    let ptx = crate::ptx_kernels::ep_token_gather_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ep_token_gather");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_buf = DeviceBuffer::<f32>::from_host(&expert_buf).expect("d_buf");
    let d_out = DeviceBuffer::<f32>::from_host(&out_init).expect("d_out");
    let d_slots = DeviceBuffer::<u32>::from_host(&slots).expect("d_slots");

    let n = n_tokens * hidden_dim;
    let block = 8_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_buf.as_device_ptr(),
                d_out.as_device_ptr(),
                d_slots.as_device_ptr(),
                n_tokens as u32,
                hidden_dim as u32,
            ),
        )
        .expect("launch ep_token_gather");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * hidden_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert_bits_eq(&out_gpu, &out_host, "ep_token_gather");
}
