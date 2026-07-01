//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` harnesses: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   `mg_restrict_kernel` ↔ [`crate::multigrid::restrict_prolong::restrict_1d`]
//!   (interior stencil), `mg_prolong_kernel` ↔
//!   [`crate::multigrid::restrict_prolong::prolong_1d`].
//! * **Independent host re-derivation** — the op is fused into larger CPU
//!   routines (the FDM/GS sweeps, CSR mat-vec and the CG inner loop run over
//!   `f64` host buffers with different blocking), so the oracle is an
//!   independent Rust re-implementation of the kernel's *documented* `f32`
//!   arithmetic: `fdm_stencil_5pt`, `gauss_seidel_step`, `csr_spmv`,
//!   `cg_axpy_dot`. These still genuinely fail if ptxas miscompiles or the PTX
//!   has a wrong constant / shift / index, because the host code is independent
//!   of the JIT-compiled PTX.
//! * **Crate oracle** — `fem_assemble_kernel` now performs the full
//!   unconstrained dense P1 stiffness assembly: per element it builds the 3x3
//!   local stiffness `K_ij = (1/(4*Area))*(b_i*b_j + c_i*c_j)` and atomically
//!   scatters it into the dense row-major `k_global[node_i*n_nodes + node_j]`.
//!   The test validates it against the crate's own
//!   [`crate::fem::p1_triangle::p1_local_stiffness`] scattered exactly as
//!   [`crate::fem::mass_stiffness::assemble_mass_stiffness`] does (no boundary
//!   elimination), within FP32 tolerance.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

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
    sm: u32,
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
    let sm = (major * 10 + minor) as u32;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length slices.
fn worst_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
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

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A failure here means ptxas rejected the hand-written PTX (a real bug) or the
/// entry name is wrong — both are hard test failures, never silently skipped.
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

/// Deterministic `f32` sample in `[0, 1)` from the crate LCG (no `next_f32`
/// exists on this crate's `LcgRng`, so we narrow its 53-bit `next_f64`).
fn next_f32(rng: &mut LcgRng) -> f32 {
    rng.next_f64() as f32
}

// ===========================================================================
// 1. fdm_stencil_5pt  —  INDEPENDENT HOST RE-DERIVATION (2D 5-point Laplacian)
// ===========================================================================
//
// out[i,j] = a*u[i,j] + b*(u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]) on interior
// cells; boundary cells are left at the output buffer's initial value. The
// kernel maps i (row, size nx) to ctaid.y/tid.y and j (col, size ny) to
// ctaid.x/tid.x, with idx = i*ny + j.

#[test]
fn fdm_stencil_5pt_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let nx = 8_usize;
    let ny = 8_usize;
    let a = 0.5_f32;
    let b = 0.25_f32;

    let mut rng = LcgRng::new(0x0FD3_5701);
    let u: Vec<f32> = (0..nx * ny)
        .map(|_| next_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let out_init = vec![0.0_f32; nx * ny];

    // Independent host re-derivation: interior cells get the stencil, boundary
    // cells stay at the init value (the kernel never writes them).
    let mut out_host = out_init.clone();
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            let idx = i * ny + j;
            let centre = u[idx];
            let neigh = u[idx - ny] + u[idx + ny] + u[idx - 1] + u[idx + 1];
            out_host[idx] = a * centre + b * neigh;
        }
    }

    let ptx = crate::ptx_kernels::fdm_stencil_5pt_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fdm_stencil_5pt_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_u = DeviceBuffer::<f32>::from_host(&u).expect("d_u");
    let d_out = DeviceBuffer::<f32>::from_host(&out_init).expect("d_out");

    // Block (16,16); grid.x covers j (ny), grid.y covers i (nx).
    let block = (16_u32, 16_u32);
    let grid = (grid_1d(ny as u32, 16), grid_1d(nx as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_u.as_device_ptr(),
                d_out.as_device_ptr(),
                nx as u32,
                ny as u32,
                a,
                b,
            ),
        )
        .expect("launch fdm_stencil_5pt_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; nx * ny];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-5, 1e-6),
            "fdm_stencil out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 2. gauss_seidel_step  —  INDEPENDENT HOST RE-DERIVATION (red-black GS sweep)
// ===========================================================================
//
// For interior cells with (i+j)%2 == color:
//   u[i,j] = 0.25 * (h2*f[i,j] + u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]).
// Within one colour the four neighbours are the opposite colour and are not
// updated, so the in-place GPU write equals reading from the original buffer.

fn run_gauss_seidel_case(fx: &GpuFixture, color: u32) {
    let nx = 8_usize;
    let ny = 8_usize;
    let h2 = 0.01_f32;

    let mut rng = LcgRng::new(0x6A5E_1DE1 ^ u64::from(color));
    let u0: Vec<f32> = (0..nx * ny)
        .map(|_| next_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let f: Vec<f32> = (0..nx * ny)
        .map(|_| next_f32(&mut rng) * 2.0 - 1.0)
        .collect();

    // Independent host re-derivation: clone, then update only matching-colour
    // interior cells, reading neighbours from the original (unchanged) buffer.
    let mut u_host = u0.clone();
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            if (i + j) as u32 % 2 != color {
                continue;
            }
            let idx = i * ny + j;
            let neigh = u0[idx - ny] + u0[idx + ny] + u0[idx - 1] + u0[idx + 1];
            u_host[idx] = 0.25 * (h2 * f[idx] + neigh);
        }
    }

    let ptx = crate::ptx_kernels::gauss_seidel_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gauss_seidel_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_u = DeviceBuffer::<f32>::from_host(&u0).expect("d_u");
    let d_f = DeviceBuffer::<f32>::from_host(&f).expect("d_f");

    let block = (16_u32, 16_u32);
    let grid = (grid_1d(ny as u32, 16), grid_1d(nx as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_u.as_device_ptr(),
                d_f.as_device_ptr(),
                nx as u32,
                ny as u32,
                h2,
                color,
            ),
        )
        .expect("launch gauss_seidel_step_kernel");
    stream.synchronize().expect("sync");

    let mut u_gpu = vec![0.0_f32; nx * ny];
    d_u.copy_to_host(&mut u_gpu).expect("copy u");

    let (rel, abs) = worst_diff(&u_gpu, &u_host);
    for k in 0..u_gpu.len() {
        assert!(
            close(u_gpu[k], u_host[k], 1e-5, 1e-6),
            "gauss_seidel color {color} u[{k}] mismatch: gpu={} host={} \
             (worst rel={rel:e} abs={abs:e})",
            u_gpu[k],
            u_host[k]
        );
    }
}

#[test]
fn gauss_seidel_step_red_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_gauss_seidel_case(&fx, 0);
}

#[test]
fn gauss_seidel_step_black_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_gauss_seidel_case(&fx, 1);
}

// ===========================================================================
// 3. csr_spmv  —  INDEPENDENT HOST RE-DERIVATION (CSR sparse mat-vec, one row/thread)
// ===========================================================================
//
// y[i] = Σ_{k ∈ row_ptr[i]..row_ptr[i+1]} val[k] * x[col[k]].

#[test]
fn csr_spmv_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Tridiagonal-ish CSR matrix with variable row lengths (rows 0 and n-1 are
    // short), so the per-row loop bounds are genuinely exercised.
    let n = 6_usize;
    let mut row_ptr = vec![0_u32];
    let mut col = Vec::<u32>::new();
    let mut val = Vec::<f32>::new();
    let mut rng = LcgRng::new(0xC5_5009);
    for i in 0..n {
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(n);
        for j in lo..hi {
            col.push(j as u32);
            val.push(next_f32(&mut rng) * 2.0 - 1.0);
        }
        row_ptr.push(col.len() as u32);
    }
    let x: Vec<f32> = (0..n).map(|_| next_f32(&mut rng) * 2.0 - 1.0).collect();

    // Independent host re-derivation in the kernel's accumulation order.
    let mut y_host = vec![0.0_f32; n];
    for i in 0..n {
        let mut acc = 0.0_f32;
        for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
            acc += val[k] * x[col[k] as usize];
        }
        y_host[i] = acc;
    }

    let ptx = crate::ptx_kernels::csr_spmv_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "csr_spmv_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col).expect("d_col");
    let d_val = DeviceBuffer::<f32>::from_host(&val).expect("d_val");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_val.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch csr_spmv_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    // The GPU fuses each term with `fma.rn`; the host uses mul+add. Over ≤3
    // terms per row the divergence is a few ulp; 1e-5 is comfortable.
    let (rel, abs) = worst_diff(&y_gpu, &y_host);
    for k in 0..y_gpu.len() {
        assert!(
            close(y_gpu[k], y_host[k], 1e-5, 1e-6),
            "csr_spmv y[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[k],
            y_host[k]
        );
    }
}

// ===========================================================================
// 4. cg_axpy_dot  —  INDEPENDENT HOST RE-DERIVATION (fused AXPY + per-element product)
// ===========================================================================
//
// x[i] <- x[i] + alpha*p[i]  (in place);  partial[i] = x_new[i] * r[i].

#[test]
fn cg_axpy_dot_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let alpha = 0.3_f32;

    let mut rng = LcgRng::new(0xC6_A89D);
    let x0: Vec<f32> = (0..n).map(|_| next_f32(&mut rng) * 2.0 - 1.0).collect();
    let p: Vec<f32> = (0..n).map(|_| next_f32(&mut rng) * 2.0 - 1.0).collect();
    let r: Vec<f32> = (0..n).map(|_| next_f32(&mut rng) * 2.0 - 1.0).collect();

    // Independent host re-derivation.
    let mut x_host = x0.clone();
    let mut partial_host = vec![0.0_f32; n];
    for i in 0..n {
        let xn = x0[i] + alpha * p[i];
        x_host[i] = xn;
        partial_host[i] = xn * r[i];
    }

    let ptx = crate::ptx_kernels::cg_axpy_dot_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cg_axpy_dot_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x0).expect("d_x");
    let d_p = DeviceBuffer::<f32>::from_host(&p).expect("d_p");
    let d_r = DeviceBuffer::<f32>::from_host(&r).expect("d_r");
    let d_partial = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_partial");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_p.as_device_ptr(),
                d_r.as_device_ptr(),
                n as u32,
                alpha,
                d_partial.as_device_ptr(),
            ),
        )
        .expect("launch cg_axpy_dot_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    let mut partial_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");
    d_partial
        .copy_to_host(&mut partial_gpu)
        .expect("copy partial");

    // x uses a single-rounding `fma.rn(alpha,p,x)` vs host two-rounding add;
    // partial is one extra multiply. ~1 ulp; 1e-5 relative is generous.
    let (rel_x, abs_x) = worst_diff(&x_gpu, &x_host);
    for k in 0..n {
        assert!(
            close(x_gpu[k], x_host[k], 1e-5, 1e-6),
            "cg_axpy x[{k}] mismatch: gpu={} host={} (worst rel={rel_x:e} abs={abs_x:e})",
            x_gpu[k],
            x_host[k]
        );
    }
    let (rel_p, abs_p) = worst_diff(&partial_gpu, &partial_host);
    for k in 0..n {
        assert!(
            close(partial_gpu[k], partial_host[k], 1e-5, 1e-6),
            "cg_axpy partial[{k}] mismatch: gpu={} host={} (worst rel={rel_p:e} abs={abs_p:e})",
            partial_gpu[k],
            partial_host[k]
        );
    }
}

// ===========================================================================
// 5. fem_assemble  —  CRATE ORACLE (full unconstrained dense P1 stiffness)
// ===========================================================================
//
// `fem_assemble_kernel` builds, per element, the full 3x3 local stiffness
//   K_ij = (1/(4*Area)) * (b_i*b_j + c_i*c_j)
// with b0=y1-y2, b1=y2-y0, b2=y0-y1 and c0=x2-x1, c1=x0-x2, c2=x1-x0, and
// atomically scatters it into the dense row-major n_nodes x n_nodes global
// matrix at k_global[node_i*n_nodes + node_j]. The oracle is the crate's own
// `p1_local_stiffness` scattered with the exact dense map used by
// `assemble_mass_stiffness` (no boundary elimination). Several nodes are shared
// across elements, so the `atom.global.add.f32` accumulation path is genuinely
// exercised on the shared (i,j) entries.

#[test]
fn fem_assemble_dense_stiffness_matches_crate() {
    use crate::fem::p1_triangle::p1_local_stiffness;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // 5 nodes, 4 elements; many (node_i, node_j) pairs receive contributions
    // from more than one element, accumulating through the atomic adds.
    let n_nodes = 5_usize;
    let coords: Vec<f32> = vec![
        0.0, 0.0, // node 0
        1.0, 0.0, // node 1
        0.0, 1.0, // node 2
        1.0, 1.0, // node 3
        2.0, 0.5, // node 4
    ];
    let conn: Vec<u32> = vec![
        0, 1, 2, // elem 0
        0, 3, 2, // elem 1
        1, 2, 4, // elem 2
        3, 4, 0, // elem 3
    ];
    let n_elem = conn.len() / 3;

    // Crate oracle: assemble the full dense stiffness exactly as
    // `assemble_mass_stiffness` does (p1_local_stiffness + dense scatter, no
    // boundary elimination), computed in f64 then narrowed to f32 for the
    // device comparison.
    let mut k_host = vec![0.0_f32; n_nodes * n_nodes];
    for e in 0..n_elem {
        let n0 = conn[3 * e] as usize;
        let n1 = conn[3 * e + 1] as usize;
        let n2 = conn[3 * e + 2] as usize;
        let x0 = f64::from(coords[2 * n0]);
        let y0 = f64::from(coords[2 * n0 + 1]);
        let x1 = f64::from(coords[2 * n1]);
        let y1 = f64::from(coords[2 * n1 + 1]);
        let x2 = f64::from(coords[2 * n2]);
        let y2 = f64::from(coords[2 * n2 + 1]);
        let k_local = p1_local_stiffness(x0, y0, x1, y1, x2, y2).expect("p1_local_stiffness");
        let idx = [n0, n1, n2];
        for i in 0..3 {
            for j in 0..3 {
                k_host[idx[i] * n_nodes + idx[j]] += k_local[i * 3 + j] as f32;
            }
        }
    }

    let ptx = crate::ptx_kernels::fem_assemble_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fem_assemble_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_coords = DeviceBuffer::<f32>::from_host(&coords).expect("d_coords");
    let d_conn = DeviceBuffer::<u32>::from_host(&conn).expect("d_conn");
    let d_k = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_nodes * n_nodes]).expect("d_k");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_elem as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_coords.as_device_ptr(),
                d_conn.as_device_ptr(),
                d_k.as_device_ptr(),
                n_elem as u32,
                n_nodes as u32,
            ),
        )
        .expect("launch fem_assemble_kernel");
    stream.synchronize().expect("sync");

    let mut k_gpu = vec![0.0_f32; n_nodes * n_nodes];
    d_k.copy_to_host(&mut k_gpu).expect("copy k");

    let (rel, abs) = worst_diff(&k_gpu, &k_host);
    for k in 0..n_nodes * n_nodes {
        assert!(
            close(k_gpu[k], k_host[k], 1e-4, 1e-5),
            "fem_assemble k_global[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            k_gpu[k],
            k_host[k]
        );
    }
}

// ===========================================================================
// 6. mg_restrict  —  CRATE ORACLE (multigrid::restrict_prolong::restrict_1d, interior)
// ===========================================================================
//
// coarse[i] = 0.25*fine[2i-1] + 0.5*fine[2i] + 0.25*fine[2i+1] for interior i;
// the kernel leaves the two boundary coarse cells (i==0, i==n_coarse-1) at their
// init value, while the crate `restrict_1d` additionally copies the fine
// boundary into them — so we compare interior cells to the crate reference and
// assert the boundary cells stayed at their init (0).

#[test]
fn mg_restrict_matches_crate_interior() {
    use crate::multigrid::restrict_prolong::restrict_1d;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_coarse = 5_usize;
    let n_fine = 2 * n_coarse - 1; // 9

    let mut rng = LcgRng::new(0x_3E57_4137);
    let fine_f32: Vec<f32> = (0..n_fine)
        .map(|_| next_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let fine_f64: Vec<f64> = fine_f32.iter().map(|&v| f64::from(v)).collect();

    // Crate oracle (f64); interior cells match the kernel's stencil.
    let coarse_ref = restrict_1d(&fine_f64).expect("restrict_1d");

    let ptx = crate::ptx_kernels::mg_restrict_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mg_restrict_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_fine = DeviceBuffer::<f32>::from_host(&fine_f32).expect("d_fine");
    let d_coarse = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_coarse]).expect("d_coarse");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_coarse as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_fine.as_device_ptr(),
                d_coarse.as_device_ptr(),
                n_coarse as u32,
            ),
        )
        .expect("launch mg_restrict_kernel");
    stream.synchronize().expect("sync");

    let mut coarse_gpu = vec![0.0_f32; n_coarse];
    d_coarse.copy_to_host(&mut coarse_gpu).expect("copy coarse");

    // Boundary cells must remain untouched (init 0).
    assert_eq!(
        coarse_gpu[0].to_bits(),
        0_u32,
        "mg_restrict: boundary coarse[0] should stay at init, got {}",
        coarse_gpu[0]
    );
    assert_eq!(
        coarse_gpu[n_coarse - 1].to_bits(),
        0_u32,
        "mg_restrict: boundary coarse[{}] should stay at init, got {}",
        n_coarse - 1,
        coarse_gpu[n_coarse - 1]
    );
    // Interior cells match the crate reference.
    for i in 1..n_coarse - 1 {
        let want = coarse_ref[i] as f32;
        assert!(
            close(coarse_gpu[i], want, 1e-5, 1e-6),
            "mg_restrict coarse[{i}] mismatch: gpu={} crate={want}",
            coarse_gpu[i]
        );
    }
}

// ===========================================================================
// 7. mg_prolong  —  CRATE ORACLE (multigrid::restrict_prolong::prolong_1d, all cells)
// ===========================================================================
//
// fine[2i] = coarse[i]; fine[2i+1] = 0.5*(coarse[i] + coarse[i+1]).

#[test]
fn mg_prolong_matches_crate() {
    use crate::multigrid::restrict_prolong::prolong_1d;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_coarse = 5_usize;
    let n_fine = 2 * n_coarse - 1; // 9

    let mut rng = LcgRng::new(0x_9201_BBCD);
    let coarse_f32: Vec<f32> = (0..n_coarse)
        .map(|_| next_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let coarse_f64: Vec<f64> = coarse_f32.iter().map(|&v| f64::from(v)).collect();

    // Crate oracle (f64): the kernel writes every fine cell, so all match.
    let fine_ref = prolong_1d(&coarse_f64).expect("prolong_1d");

    let ptx = crate::ptx_kernels::mg_prolong_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mg_prolong_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_coarse = DeviceBuffer::<f32>::from_host(&coarse_f32).expect("d_coarse");
    let d_fine = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_fine]).expect("d_fine");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_fine as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_coarse.as_device_ptr(),
                d_fine.as_device_ptr(),
                n_fine as u32,
            ),
        )
        .expect("launch mg_prolong_kernel");
    stream.synchronize().expect("sync");

    let mut fine_gpu = vec![0.0_f32; n_fine];
    d_fine.copy_to_host(&mut fine_gpu).expect("copy fine");

    for i in 0..n_fine {
        let want = fine_ref[i] as f32;
        assert!(
            close(fine_gpu[i], want, 1e-5, 1e-6),
            "mg_prolong fine[{i}] mismatch: gpu={} crate={want}",
            fine_gpu[i]
        );
    }
}
