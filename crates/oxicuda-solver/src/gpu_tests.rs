//! On-device GPU validation for the hand-written PTX kernels emitted by
//! `oxicuda-solver`.
//!
//! Unlike the domain crates, this crate emits PTX through generator functions
//! (`emit_*`, `generate_*_ptx`) and `*Plan::generate_ptx` builders rather than a
//! single `ptx_kernels.rs`. Each test here JIT-compiles a kernel's PTX for the
//! live device via `Module::from_ptx`, launches it on the real CUDA device
//! through `oxicuda-launch`, copies the results back, and asserts numerical
//! equivalence to an independent CPU re-derivation of the kernel's arithmetic.
//!
//! The launch ABI mirrors the `oxicuda-cs` / `oxicuda-ot` canaries: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar (`.param .u32`), in the kernel's declared parameter
//! order.
//!
//! ## Kernel inventory (honest accounting)
//!
//! * **Validated against a CPU oracle** — LU (`trsm_unit_lower`, `gemm_update`,
//!   `panel_lu`, `pivot_swap`), Cholesky (`chol_panel_trsm`, `chol_syrk`,
//!   `panel_cholesky`, both triangles), the row-swap helper, and all ten
//!   `matrix_functions` kernels (expm scale/pade/square, logm
//!   shift/sqrt_step/pade/scale_back, sqrtm init/iter/conv).
//! * **JIT-load + launch only (no oracle)** — the batched kernels
//!   (`batched_lu/qr/cholesky/solve`) and QZ kernels
//!   (`hessenberg_reduction/qz_sweep/eigenvalue_extract`) are deliberate
//!   placeholder bodies (they compute offsets then `ret` without writing
//!   results). They assemble and launch cleanly but have no numerical output,
//!   so they are load-validated only.
//!
//! Every test skips (returns early) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_blas::types::FillMode;
use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;

// ---------------------------------------------------------------------------
// Fixture & shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
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
    let dev = Device::get(0).ok()?;
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = sm_from_cc(major, minor);
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// Maps a numeric compute capability to the nearest supported [`SmVersion`].
fn sm_from_cc(major: i32, minor: i32) -> SmVersion {
    SmVersion::from_compute_capability(major, minor).unwrap_or(SmVersion::Sm80)
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in
/// the kernel string, surfaced as a test failure rather than silently skipped.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx).unwrap_or_else(|e| {
        panic!("PTX JIT compile failed for `{entry}`: {e}\n--- PTX ---\n{ptx}")
    });
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a 1-D grid size (at least 1).
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block).max(1)
}

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Asserts two FP32 slices agree within tolerance, reporting the first mismatch.
fn assert_close_slice(gpu: &[f32], cpu: &[f32], rel: f32, abs: f32, tag: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{tag}: length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close(g, c, rel, abs),
            "{tag}: element {i} mismatch gpu={g} cpu={c} (rel={rel:e} abs={abs:e})"
        );
    }
}

/// A small deterministic LCG. Normalisation divides by `2^32` (never `2^31`).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform in `[0, 1)` via division by `2^32`.
    fn unit(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
    /// Uniform `f32` in `[lo, hi)`.
    fn range_f32(&mut self, lo: f64, hi: f64) -> f32 {
        (lo + (hi - lo) * self.unit()) as f32
    }
}

/// Splits a string holding several concatenated PTX modules (each beginning with
/// a `.version` line, as produced by joining `KernelBuilder::build` outputs) into
/// individual standalone modules.
fn split_modules(joined: &str) -> Vec<String> {
    let mut modules: Vec<String> = Vec::new();
    let mut current = String::new();
    for line in joined.lines() {
        if line.starts_with(".version") && !current.trim().is_empty() {
            modules.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        modules.push(current);
    }
    modules
}

/// Extracts the `.visible .entry NAME(` identifier from a single PTX module.
fn entry_name(module: &str) -> String {
    let marker = ".visible .entry ";
    let start = module
        .find(marker)
        .map(|p| p + marker.len())
        .expect("module must contain a .visible .entry");
    let rest = &module[start..];
    let end = rest
        .find(|c: char| c == '(' || c.is_whitespace())
        .unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

/// Returns the standalone module containing an entry whose name contains
/// `needle`, together with that entry's name.
fn module_with_entry(joined: &str, needle: &str) -> (String, String) {
    for m in split_modules(joined) {
        let name = entry_name(&m);
        if name.contains(needle) {
            return (m, name);
        }
    }
    panic!("no module with entry containing `{needle}` in joined PTX");
}

/// Best-effort `ptxas` pre-screen: assembles `ptx` for its own declared
/// `.target`. Returns `Ok(())` on success or when `ptxas` is unavailable, and
/// the captured stderr on assembler rejection.
fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::Command;

    let ptxas = "/usr/local/cuda/bin/ptxas";
    if !std::path::Path::new(ptxas).exists() {
        return Ok(());
    }
    // Use the module's own declared target to avoid arch-mismatch false errors.
    let arch = ptx
        .lines()
        .find_map(|l| l.trim().strip_prefix(".target "))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "sm_86".to_string());

    let dir = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let in_path = dir.join(format!("oxisolver_{tag}_{stamp}.ptx"));
    let out_path = dir.join(format!("oxisolver_{tag}_{stamp}.cubin"));

    {
        let mut f = match std::fs::File::create(&in_path) {
            Ok(f) => f,
            Err(_) => return Ok(()),
        };
        if f.write_all(ptx.as_bytes()).is_err() {
            return Ok(());
        }
    }

    let output = Command::new(ptxas)
        .arg(format!("-arch={arch}"))
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output();

    let result = match output {
        Ok(o) if o.status.success() => Ok(()),
        Ok(o) => Err(format!(
            "ptxas rejected `{tag}` (arch={arch}): {}",
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(_) => Ok(()),
    };

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    result
}

// ===========================================================================
// 0. ptxas pre-screen of every emitter (no GPU required)
// ===========================================================================

#[test]
fn ptxas_prescreen_all_kernels() {
    let sm = SmVersion::Sm86;
    let mut failures: Vec<String> = Vec::new();

    let mut check = |ptx: &str, tag: &str| {
        if let Err(e) = ptxas_assembles(ptx, tag) {
            failures.push(e);
        }
    };

    // LU.
    check(
        &crate::dense::lu::emit_trsm_unit_lower::<f32>(sm).expect("trsm"),
        "lu_trsm",
    );
    check(
        &crate::dense::lu::emit_gemm_update::<f32>(sm).expect("gemm"),
        "lu_gemm",
    );
    check(
        &crate::dense::lu::emit_panel_lu::<f32>(sm, 4).expect("panel_lu"),
        "lu_panel",
    );
    check(
        &crate::dense::lu::emit_pivot_swap::<f32>(sm).expect("pivot_swap"),
        "lu_pivot_swap",
    );
    // Cholesky (both triangles).
    for lower in [true, false] {
        check(
            &crate::dense::cholesky::emit_chol_panel_trsm::<f32>(sm, lower).expect("chol_trsm"),
            "chol_trsm",
        );
        check(
            &crate::dense::cholesky::emit_chol_syrk::<f32>(sm, lower).expect("chol_syrk"),
            "chol_syrk",
        );
        let fm = if lower {
            FillMode::Lower
        } else {
            FillMode::Upper
        };
        check(
            &crate::dense::cholesky::emit_panel_cholesky::<f32>(sm, 4, fm).expect("panel_chol"),
            "panel_chol",
        );
    }
    // Row-swap helper.
    check(
        &crate::helpers::pivot::generate_row_swap_ptx::<f32>(sm).expect("row_swap"),
        "row_swap",
    );
    // matrix_functions (each Plan emits a joined multi-module string).
    let expm = crate::dense::matrix_functions::MatrixExpPlan::new(
        crate::dense::matrix_functions::MatrixExpConfig::new(4, "f32").with_pade_order(5),
    )
    .expect("expm plan")
    .generate_ptx()
    .expect("expm ptx");
    for m in split_modules(&expm) {
        let n = entry_name(&m);
        check(&m, &n);
    }
    let logm = crate::dense::matrix_functions::MatrixLogPlan::new(
        crate::dense::matrix_functions::MatrixLogConfig::new(4, "f32"),
    )
    .expect("logm plan")
    .generate_ptx()
    .expect("logm ptx");
    for m in split_modules(&logm) {
        let n = entry_name(&m);
        check(&m, &n);
    }
    let sqrtm = crate::dense::matrix_functions::MatrixSqrtPlan::new(
        crate::dense::matrix_functions::MatrixSqrtConfig::new(4, "f32"),
    )
    .expect("sqrtm plan")
    .generate_ptx()
    .expect("sqrtm ptx");
    for m in split_modules(&sqrtm) {
        let n = entry_name(&m);
        check(&m, &n);
    }
    // Stub kernels (batched + QZ) must still assemble.
    check(
        &crate::dense::batched::emit_batched_lu::<f32>(sm, 16).expect("blu"),
        "batched_lu",
    );
    check(
        &crate::dense::batched::emit_batched_qr::<f32>(sm, 16, 8).expect("bqr"),
        "batched_qr",
    );
    check(
        &crate::dense::batched::emit_batched_cholesky::<f32>(sm, 16).expect("bchol"),
        "batched_cholesky",
    );
    check(
        &crate::dense::batched::emit_batched_solve::<f32>(sm, 16, 4).expect("bsolve"),
        "batched_solve",
    );
    check(
        &crate::dense::qz::generate_hessenberg_reduction_ptx(8, sm).expect("qz_hess"),
        "qz_hessenberg",
    );
    check(
        &crate::dense::qz::generate_qz_sweep_ptx(8, sm).expect("qz_sweep"),
        "qz_sweep",
    );
    check(
        &crate::dense::qz::generate_eigenvalue_extraction_ptx(8, sm).expect("qz_eig"),
        "qz_eig",
    );

    assert!(
        failures.is_empty(),
        "ptxas rejected {} kernel(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ===========================================================================
// LU kernels
// ===========================================================================

#[test]
fn lu_trsm_unit_lower_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    let jb = 4usize;
    let rcols = 3usize;
    let ldl = jb;
    let ldb = jb;
    let mut rng = Lcg::new(0x1234_5678);

    // Unit-lower L (column-major): only strictly-lower entries are read.
    let mut l = vec![0.0f32; ldl * jb];
    for col in 0..jb {
        for row in 0..jb {
            l[row + col * ldl] = if row == col {
                1.0
            } else if row > col {
                rng.range_f32(-0.8, 0.8)
            } else {
                0.0
            };
        }
    }
    let mut b_host = vec![0.0f32; ldb * rcols];
    for v in b_host.iter_mut() {
        *v = rng.range_f32(-2.0, 2.0);
    }

    // CPU oracle: column-wise forward substitution.
    let mut expect = b_host.clone();
    for c in 0..rcols {
        for i in 0..jb {
            let mut acc = expect[i + c * ldb];
            for k in 0..i {
                acc -= l[i + k * ldl] * expect[k + c * ldb];
            }
            expect[i + c * ldb] = acc;
        }
    }

    let ptx = crate::dense::lu::emit_trsm_unit_lower::<f32>(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "solver_lu_trsm_ll_f32");
    let d_l = DeviceBuffer::<f32>::from_host(&l).expect("d_l");
    let d_b = DeviceBuffer::<f32>::from_host(&b_host).expect("d_b");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d(rcols as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_l.as_device_ptr(),
                d_b.as_device_ptr(),
                jb as u32,
                rcols as u32,
                ldl as u32,
                ldb as u32,
            ),
        )
        .expect("launch trsm");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; ldb * rcols];
    d_b.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 1e-4, 1e-5, "lu_trsm");
}

#[test]
fn lu_gemm_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    // Non-square trailing update to exercise the lda/ldb/ldc index math.
    let rrows = 5usize;
    let rcols = 7usize;
    let kk = 3usize;
    let lda = rrows;
    let ldb = kk;
    let ldc = rrows;
    let mut rng = Lcg::new(0xACE1_2024);

    let a: Vec<f32> = (0..lda * kk).map(|_| rng.range_f32(-1.5, 1.5)).collect();
    let bmat: Vec<f32> = (0..ldb * rcols).map(|_| rng.range_f32(-1.5, 1.5)).collect();
    let c0: Vec<f32> = (0..ldc * rcols).map(|_| rng.range_f32(-1.5, 1.5)).collect();

    // CPU oracle: C := C - A*B with fma accumulation (matches device fma.rn).
    let mut expect = c0.clone();
    for col in 0..rcols {
        for row in 0..rrows {
            let mut acc = 0.0f32;
            for k in 0..kk {
                acc = a[row + k * lda].mul_add(bmat[k + col * ldb], acc);
            }
            expect[row + col * ldc] -= acc;
        }
    }

    let ptx = crate::dense::lu::emit_gemm_update::<f32>(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "solver_lu_gemm_update_f32");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&bmat).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c0).expect("d_c");

    const TILE: u32 = 16;
    // Kernel maps row<-Y dim, col<-X dim: X must cover rcols, Y must cover rrows.
    let grid = (grid_1d(rcols as u32, TILE), grid_1d(rrows as u32, TILE));
    let params = LaunchParams::new(grid, (TILE, TILE));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                rrows as u32,
                rcols as u32,
                kk as u32,
                lda as u32,
                ldb as u32,
                ldc as u32,
            ),
        )
        .expect("launch gemm");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; ldc * rcols];
    d_c.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 1e-4, 1e-5, "lu_gemm");
}

#[test]
fn lu_panel_lu_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    let m = 6usize;
    let nc = 4usize;
    let lda = m;
    let j = 0u32;
    let mut rng = Lcg::new(0x5EED_F00D);

    // A diagonally biased panel keeps every pivot comfortably non-zero.
    let mut panel = vec![0.0f32; lda * nc];
    for col in 0..nc {
        for row in 0..m {
            let mut v = rng.range_f32(-1.0, 1.0);
            if row == col {
                v += 5.0;
            }
            panel[row + col * lda] = v;
        }
    }

    // CPU oracle: right-looking Doolittle LU with partial pivoting.
    let mut expect = panel.clone();
    let mut piv_expect = vec![0u32; nc];
    for k in 0..nc {
        let mut maxabs = 0.0f32;
        let mut prow = k;
        for r in k..m {
            let av = expect[r + k * lda].abs();
            if av > maxabs {
                maxabs = av;
                prow = r;
            }
        }
        piv_expect[k] = j + prow as u32;
        if prow != k {
            for c in 0..nc {
                expect.swap(k + c * lda, prow + c * lda);
            }
        }
        let pivot = expect[k + k * lda];
        for r in k + 1..m {
            expect[r + k * lda] /= pivot;
        }
        for t in k + 1..nc {
            let akt = expect[k + t * lda];
            for r in k + 1..m {
                let prod = expect[r + k * lda] * akt;
                expect[r + t * lda] -= prod;
            }
        }
    }

    let ptx = crate::dense::lu::emit_panel_lu::<f32>(fx.sm, nc as u32).expect("ptx");
    let kernel = load_kernel(&ptx, &format!("solver_panel_lu_f32_{nc}"));
    let d_panel = DeviceBuffer::<f32>::from_host(&panel).expect("d_panel");
    let d_piv = DeviceBuffer::<u32>::from_host(&vec![0u32; nc]).expect("d_piv");
    let info_init = vec![999u32];
    let d_info = DeviceBuffer::<u32>::from_host(&info_init).expect("d_info");

    let params = LaunchParams::new(1u32, nc as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_panel.as_device_ptr(),
                d_piv.as_device_ptr(),
                d_info.as_device_ptr(),
                m as u32,
                nc as u32,
                j,
                lda as u32,
            ),
        )
        .expect("launch panel_lu");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; lda * nc];
    d_panel.copy_to_host(&mut got).expect("copy panel");
    let mut piv_got = vec![0u32; nc];
    d_piv.copy_to_host(&mut piv_got).expect("copy piv");
    let mut info_got = vec![0u32; 1];
    d_info.copy_to_host(&mut info_got).expect("copy info");

    assert_eq!(piv_got, piv_expect, "panel_lu pivots");
    assert_eq!(
        info_got[0], 999,
        "panel_lu info must stay unset (non-singular)"
    );
    assert_close_slice(&got, &expect, 1e-4, 1e-5, "lu_panel");
}

#[test]
fn lu_pivot_swap_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    let m = 5usize;
    let cols = 3usize;
    let lda = m;
    let j = 0u32;
    let jb = 3u32;
    let mut rng = Lcg::new(0xBEEF_CAFE);

    let a: Vec<f32> = (0..lda * cols).map(|_| rng.range_f32(-3.0, 3.0)).collect();
    // Transposition sequence pivots[j+t] (absolute row indices).
    let pivots = vec![2u32, 2u32, 4u32];

    // CPU oracle: replay transpositions column by column.
    let mut expect = a.clone();
    for c in 0..cols {
        let base = c * lda;
        for t in 0..jb as usize {
            let row = j as usize + t;
            let p = pivots[row] as usize;
            if p != row {
                expect.swap(base + row, base + p);
            }
        }
    }

    let ptx = crate::dense::lu::emit_pivot_swap::<f32>(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "solver_pivot_swap_f32");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_piv = DeviceBuffer::<u32>::from_host(&pivots).expect("d_piv");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d(cols as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_piv.as_device_ptr(),
                j,
                jb,
                0u32,
                cols as u32,
                lda as u32,
            ),
        )
        .expect("launch pivot_swap");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; lda * cols];
    d_a.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 0.0, 0.0, "lu_pivot_swap");
}

// ===========================================================================
// Cholesky kernels (both triangles)
// ===========================================================================

#[test]
fn chol_panel_trsm_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    for is_lower in [true, false] {
        let free = 4usize;
        let jb = 3usize;
        let ldt = jb;
        let mut rng = Lcg::new(if is_lower { 0xC401 } else { 0xC402 });

        // Triangular diagonal block D (jb x jb) with comfortably non-zero diagonal.
        let mut d = vec![0.0f32; ldt * jb];
        for col in 0..jb {
            for row in 0..jb {
                let on_used_triangle = if is_lower { row >= col } else { row <= col };
                let mut v = if on_used_triangle {
                    rng.range_f32(-0.6, 0.6)
                } else {
                    0.0
                };
                if row == col {
                    v = 2.0 + rng.range_f32(0.0, 1.0);
                }
                d[row + col * ldt] = v;
            }
        }

        // Panel P. Lower: free x jb (lda=free). Upper: jb x free (lda=jb).
        let lda = if is_lower { free } else { jb };
        let prows = if is_lower { free } else { jb };
        let pcols = if is_lower { jb } else { free };
        let mut p = vec![0.0f32; lda * pcols];
        for c in 0..pcols {
            for r in 0..prows {
                p[r + c * lda] = rng.range_f32(-1.5, 1.5);
            }
        }

        // CPU oracle mirroring the kernel's index mapping.
        let mut expect = p.clone();
        for g in 0..free {
            for pp in 0..jb {
                let (solve_idx, diag_idx) = if is_lower {
                    (g + pp * lda, pp + pp * ldt)
                } else {
                    (pp + g * lda, pp + pp * ldt)
                };
                let mut acc = expect[solve_idx];
                for k in 0..pp {
                    let (tri_idx, x_idx) = if is_lower {
                        (pp + k * ldt, g + k * lda)
                    } else {
                        (k + pp * ldt, k + g * lda)
                    };
                    acc -= d[tri_idx] * expect[x_idx];
                }
                expect[solve_idx] = acc / d[diag_idx];
            }
        }

        let ptx =
            crate::dense::cholesky::emit_chol_panel_trsm::<f32>(fx.sm, is_lower).expect("ptx");
        let entry = if is_lower {
            "solver_chol_trsm_lower_f32"
        } else {
            "solver_chol_trsm_upper_f32"
        };
        let kernel = load_kernel(&ptx, entry);
        let d_diag = DeviceBuffer::<f32>::from_host(&d).expect("d_diag");
        let d_panel = DeviceBuffer::<f32>::from_host(&p).expect("d_panel");
        let block = 256u32;
        let params = LaunchParams::new(grid_1d(free as u32, block), block);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_diag.as_device_ptr(),
                    d_panel.as_device_ptr(),
                    jb as u32,
                    free as u32,
                    ldt as u32,
                    lda as u32,
                ),
            )
            .expect("launch chol_trsm");
        stream.synchronize().expect("sync");

        let mut got = vec![0.0f32; lda * pcols];
        d_panel.copy_to_host(&mut got).expect("copy");
        assert_close_slice(
            &got,
            &expect,
            1e-4,
            1e-5,
            if is_lower {
                "chol_trsm_lower"
            } else {
                "chol_trsm_upper"
            },
        );
    }
}

#[test]
fn chol_syrk_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    for is_lower in [true, false] {
        let rem = 5usize;
        let jb = 3usize;
        let ldc = rem;
        let mut rng = Lcg::new(if is_lower { 0x5401 } else { 0x5402 });

        // Operand P. Lower: rem x jb (ldp=rem). Upper: jb x rem (ldp=jb).
        let ldp = if is_lower { rem } else { jb };
        let prows = if is_lower { rem } else { jb };
        let pcols = if is_lower { jb } else { rem };
        let p: Vec<f32> = (0..ldp * pcols).map(|_| rng.range_f32(-1.2, 1.2)).collect();
        let c0: Vec<f32> = (0..ldc * rem).map(|_| rng.range_f32(-1.2, 1.2)).collect();
        let _ = (prows, pcols);

        // CPU oracle: only the active triangle of C is updated.
        let mut expect = c0.clone();
        for col in 0..rem {
            for row in 0..rem {
                let active = if is_lower { row >= col } else { row <= col };
                if !active {
                    continue;
                }
                let mut acc = 0.0f32;
                for k in 0..jb {
                    let (li, ri) = if is_lower {
                        (row + k * ldp, col + k * ldp)
                    } else {
                        (k + row * ldp, k + col * ldp)
                    };
                    acc = p[li].mul_add(p[ri], acc);
                }
                expect[row + col * ldc] -= acc;
            }
        }

        let ptx = crate::dense::cholesky::emit_chol_syrk::<f32>(fx.sm, is_lower).expect("ptx");
        let entry = if is_lower {
            "solver_chol_syrk_lower_f32"
        } else {
            "solver_chol_syrk_upper_f32"
        };
        let kernel = load_kernel(&ptx, entry);
        let d_panel = DeviceBuffer::<f32>::from_host(&p).expect("d_panel");
        let d_c = DeviceBuffer::<f32>::from_host(&c0).expect("d_c");

        const TILE: u32 = 16;
        let grid = (grid_1d(rem as u32, TILE), grid_1d(rem as u32, TILE));
        let params = LaunchParams::new(grid, (TILE, TILE));
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_panel.as_device_ptr(),
                    d_c.as_device_ptr(),
                    rem as u32,
                    jb as u32,
                    ldp as u32,
                    ldc as u32,
                ),
            )
            .expect("launch chol_syrk");
        stream.synchronize().expect("sync");

        let mut got = vec![0.0f32; ldc * rem];
        d_c.copy_to_host(&mut got).expect("copy");
        assert_close_slice(
            &got,
            &expect,
            1e-4,
            1e-5,
            if is_lower {
                "chol_syrk_lower"
            } else {
                "chol_syrk_upper"
            },
        );
    }
}

#[test]
fn panel_cholesky_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    for is_lower in [true, false] {
        let jb = 4usize;
        let lda = jb;
        let mut rng = Lcg::new(if is_lower { 0xC4ED } else { 0xC4EE });

        // SPD matrix A = M^T M + jb*I (column-major).
        let mmat: Vec<f32> = (0..jb * jb).map(|_| rng.range_f32(-1.0, 1.0)).collect();
        let mut a = vec![0.0f32; lda * jb];
        for col in 0..jb {
            for row in 0..jb {
                let mut acc = 0.0f32;
                for k in 0..jb {
                    acc += mmat[k + row * jb] * mmat[k + col * jb];
                }
                if row == col {
                    acc += jb as f32;
                }
                a[row + col * lda] = acc;
            }
        }

        // CPU oracle mirroring the kernel's triangle-aware in-place factorization.
        let mut expect = a.clone();
        let elem = |row: usize, col: usize| -> usize {
            if is_lower {
                row + col * lda
            } else {
                col + row * lda
            }
        };
        for k in 0..jb {
            let pivot = expect[elem(k, k)].sqrt();
            expect[elem(k, k)] = pivot;
            for r in k + 1..jb {
                expect[elem(r, k)] /= pivot;
            }
            for t in k + 1..jb {
                let atk = expect[elem(t, k)];
                for r in t..jb {
                    let prod = expect[elem(r, k)] * atk;
                    expect[elem(r, t)] -= prod;
                }
            }
        }

        let fm = if is_lower {
            FillMode::Lower
        } else {
            FillMode::Upper
        };
        let ptx =
            crate::dense::cholesky::emit_panel_cholesky::<f32>(fx.sm, jb as u32, fm).expect("ptx");
        let tri = if is_lower { "lower" } else { "upper" };
        let entry = format!("solver_panel_cholesky_{tri}_f32_{jb}");
        let kernel = load_kernel(&ptx, &entry);
        let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
        let params = LaunchParams::new(1u32, jb as u32);
        kernel
            .launch(
                &params,
                &stream,
                &(d_a.as_device_ptr(), jb as u32, lda as u32),
            )
            .expect("launch panel_cholesky");
        stream.synchronize().expect("sync");

        let mut got = vec![0.0f32; lda * jb];
        d_a.copy_to_host(&mut got).expect("copy");
        // Compare only the factorized (active) triangle incl. diagonal; the
        // opposite triangle is left as the original input by both paths.
        for col in 0..jb {
            for row in 0..jb {
                let active = if is_lower { row >= col } else { row <= col };
                if active {
                    let idx = row + col * lda;
                    assert!(
                        close(got[idx], expect[idx], 1e-4, 1e-5),
                        "panel_cholesky_{tri} ({row},{col}) gpu={} cpu={}",
                        got[idx],
                        expect[idx]
                    );
                }
            }
        }
    }
}

// ===========================================================================
// Row-swap helper  (+ non-vacuous device-path probe)
// ===========================================================================

#[test]
fn row_swap_matches_host_and_is_non_vacuous() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");

    let m = 5usize;
    let n_cols = 4usize;
    let lda = m;
    let row1 = 1u32;
    let row2 = 3u32;
    let mut rng = Lcg::new(0x5A11);
    let a: Vec<f32> = (0..lda * n_cols)
        .map(|_| rng.range_f32(-9.0, 9.0))
        .collect();

    let mut expect = a.clone();
    for c in 0..n_cols {
        expect.swap(row1 as usize + c * lda, row2 as usize + c * lda);
    }

    let ptx = crate::helpers::pivot::generate_row_swap_ptx::<f32>(fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "solver_row_swap_f32");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d(n_cols as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_a.as_device_ptr(), row1, row2, n_cols as u32, lda as u32),
        )
        .expect("launch row_swap");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; lda * n_cols];
    d_a.copy_to_host(&mut got).expect("copy");

    // (a) The device result matches the host oracle exactly (pure data movement).
    assert_close_slice(&got, &expect, 0.0, 0.0, "row_swap");
    // (b) Non-vacuous: the kernel actually mutated device memory — the result
    //     differs from the un-swapped input (rows 1 and 3 are distinct).
    assert_ne!(got, a, "row_swap kernel did not change device memory");
}

// ===========================================================================
// matrix_functions kernels (expm / logm / sqrtm)
// ===========================================================================

use crate::dense::matrix_functions::{
    MatrixExpConfig, MatrixExpPlan, MatrixLogConfig, MatrixLogPlan, MatrixSqrtConfig,
    MatrixSqrtPlan,
};

/// Launches a 1-D element-wise matrix_functions kernel and returns the result.
fn run_expm_joined(needle: &str) -> (String, String) {
    let plan =
        MatrixExpPlan::new(MatrixExpConfig::new(4, "f32").with_pade_order(5)).expect("expm plan");
    let joined = plan.generate_ptx().expect("expm ptx");
    module_with_entry(&joined, needle)
}

#[test]
fn expm_scale_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let n = 4usize;
    let scale_exp = 2u32; // divide by 2^2 = 4.
    let mut rng = Lcg::new(0xE401);
    let a: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-8.0, 8.0)).collect();
    let expect: Vec<f32> = a.iter().map(|&v| v / 4.0).collect();

    let (module, entry) = run_expm_joined("expm_scale");
    let kernel = load_kernel(&module, &entry);
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_out");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                scale_exp,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; n * n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 1e-6, 0.0, "expm_scale");
}

#[test]
fn expm_pade_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let n = 4usize;
    // Coefficients are stored in the kernel's working precision (f32 here).
    let coeffs: Vec<f32> = vec![1.0, 2.0, 3.0, 0.5];
    let num_coeffs = coeffs.len() as u32;
    let mut rng = Lcg::new(0xE402);
    let a: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    // CPU oracle: scalar Horner per element (P all +, Q alternating sign).
    let mut p_exp = vec![0.0f32; n * n];
    let mut q_exp = vec![0.0f32; n * n];
    for i in 0..n * n {
        let x = a[i];
        let mut acc_p = 0.0f32;
        let mut acc_q = 0.0f32;
        for idx in (0..coeffs.len()).rev() {
            let c = coeffs[idx];
            acc_p = acc_p.mul_add(x, c);
            let qc = if idx % 2 == 1 { -c } else { c };
            acc_q = acc_q.mul_add(x, qc);
        }
        p_exp[i] = acc_p;
        q_exp[i] = acc_q;
    }

    let (module, entry) = run_expm_joined("expm_pade");
    let kernel = load_kernel(&module, &entry);
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_p = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_p");
    let d_q = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_q");
    let d_coeffs = DeviceBuffer::<f32>::from_host(&coeffs).expect("d_coeffs");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_p.as_device_ptr(),
                d_q.as_device_ptr(),
                n as u32,
                d_coeffs.as_device_ptr(),
                num_coeffs,
            ),
        )
        .expect("launch");
    stream.synchronize().expect("sync");
    let mut p_got = vec![0.0f32; n * n];
    let mut q_got = vec![0.0f32; n * n];
    d_p.copy_to_host(&mut p_got).expect("copy p");
    d_q.copy_to_host(&mut q_got).expect("copy q");
    assert_close_slice(&p_got, &p_exp, 1e-5, 1e-5, "expm_pade_P");
    assert_close_slice(&q_got, &q_exp, 1e-5, 1e-5, "expm_pade_Q");
}

#[test]
fn expm_square_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let n = 4usize;
    let mut rng = Lcg::new(0xE403);
    let f: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    // CPU oracle: tmp = F*F, column-major, fma accumulation.
    let mut expect = vec![0.0f32; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc = f[k * n + row].mul_add(f[col * n + k], acc);
            }
            expect[col * n + row] = acc;
        }
    }

    let (module, entry) = run_expm_joined("expm_square");
    let kernel = load_kernel(&module, &entry);
    let d_f = DeviceBuffer::<f32>::from_host(&f).expect("d_f");
    let d_tmp = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_tmp");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_f.as_device_ptr(), d_tmp.as_device_ptr(), n as u32),
        )
        .expect("launch");
    stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; n * n];
    d_tmp.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 1e-5, 1e-5, "expm_square");
}

#[test]
fn logm_kernels_match_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let n = 4usize;
    let plan = MatrixLogPlan::new(MatrixLogConfig::new(n as u32, "f32")).expect("logm plan");
    let joined = plan.generate_ptx().expect("logm ptx");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    let mut rng = Lcg::new(0x106E);

    // --- shift: out = A - I ---
    {
        let a: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-3.0, 3.0)).collect();
        let mut expect = a.clone();
        for d in 0..n {
            expect[d + d * n] -= 1.0;
        }
        let (module, entry) = module_with_entry(&joined, "logm_shift");
        let kernel = load_kernel(&module, &entry);
        let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
        let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_out");
        kernel
            .launch(
                &params,
                &stream,
                &(d_a.as_device_ptr(), d_out.as_device_ptr(), n as u32),
            )
            .expect("launch shift");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * n];
        d_out.copy_to_host(&mut got).expect("copy");
        assert_close_slice(&got, &expect, 1e-6, 1e-6, "logm_shift");
    }

    // --- sqrt_step: Y_next=(Y+I)/2, Z_next=(Z+I)/2 ---
    {
        let y: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
        let z: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
        let mut ye = y.clone();
        let mut ze = z.clone();
        for col in 0..n {
            for row in 0..n {
                let d = if row == col { 1.0 } else { 0.0 };
                ye[row + col * n] = (y[row + col * n] + d) * 0.5;
                ze[row + col * n] = (z[row + col * n] + d) * 0.5;
            }
        }
        let (module, entry) = module_with_entry(&joined, "logm_sqrt_step");
        let kernel = load_kernel(&module, &entry);
        let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
        let d_z = DeviceBuffer::<f32>::from_host(&z).expect("d_z");
        let d_yn = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_yn");
        let d_zn = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_zn");
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_y.as_device_ptr(),
                    d_z.as_device_ptr(),
                    d_yn.as_device_ptr(),
                    d_zn.as_device_ptr(),
                    n as u32,
                ),
            )
            .expect("launch sqrt_step");
        stream.synchronize().expect("sync");
        let mut yg = vec![0.0f32; n * n];
        let mut zg = vec![0.0f32; n * n];
        d_yn.copy_to_host(&mut yg).expect("copy y");
        d_zn.copy_to_host(&mut zg).expect("copy z");
        assert_close_slice(&yg, &ye, 1e-6, 1e-6, "logm_sqrt_step_Y");
        assert_close_slice(&zg, &ze, 1e-6, 1e-6, "logm_sqrt_step_Z");
    }

    // --- pade: truncated log(1+x) series per element ---
    {
        let num_terms = 6u32;
        let x: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-0.3, 0.3)).collect();
        let mut expect = vec![0.0f32; n * n];
        for i in 0..n * n {
            let xv = x[i];
            let mut acc = 0.0f32;
            let mut k = num_terms;
            while k >= 1 {
                let inv_k = 1.0f32 / k as f32;
                let signed = if k % 2 == 1 { inv_k } else { -inv_k };
                acc = xv.mul_add(acc, signed);
                k -= 1;
            }
            expect[i] = xv * acc;
        }
        let (module, entry) = module_with_entry(&joined, "logm_pade");
        let kernel = load_kernel(&module, &entry);
        let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
        let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_out");
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_x.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n as u32,
                    num_terms,
                ),
            )
            .expect("launch pade_log");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * n];
        d_out.copy_to_host(&mut got).expect("copy");
        assert_close_slice(&got, &expect, 1e-5, 1e-6, "logm_pade");
    }

    // --- scale_back: result *= 2^scale_exp (in place) ---
    {
        let scale_exp = 3u32; // multiply by 8.
        let r0: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
        let expect: Vec<f32> = r0.iter().map(|&v| v * 8.0).collect();
        let (module, entry) = module_with_entry(&joined, "logm_scale_back");
        let kernel = load_kernel(&module, &entry);
        let d_r = DeviceBuffer::<f32>::from_host(&r0).expect("d_r");
        kernel
            .launch(
                &params,
                &stream,
                &(d_r.as_device_ptr(), n as u32, scale_exp),
            )
            .expect("launch scale_back");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * n];
        d_r.copy_to_host(&mut got).expect("copy");
        assert_close_slice(&got, &expect, 1e-6, 0.0, "logm_scale_back");
    }
}

#[test]
fn sqrtm_kernels_match_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let n = 4usize;
    let plan = MatrixSqrtPlan::new(MatrixSqrtConfig::new(n as u32, "f32")).expect("sqrtm plan");
    let joined = plan.generate_ptx().expect("sqrtm ptx");
    let block = 256u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    let mut rng = Lcg::new(0x5417);

    // --- init: Y=A, Z=I ---
    {
        let a: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-3.0, 3.0)).collect();
        let mut z_exp = vec![0.0f32; n * n];
        for d in 0..n {
            z_exp[d + d * n] = 1.0;
        }
        let (module, entry) = module_with_entry(&joined, "sqrtm_init");
        let kernel = load_kernel(&module, &entry);
        let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
        let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_y");
        let d_z = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_z");
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_a.as_device_ptr(),
                    d_y.as_device_ptr(),
                    d_z.as_device_ptr(),
                    n as u32,
                ),
            )
            .expect("launch init");
        stream.synchronize().expect("sync");
        let mut yg = vec![0.0f32; n * n];
        let mut zg = vec![0.0f32; n * n];
        d_y.copy_to_host(&mut yg).expect("copy y");
        d_z.copy_to_host(&mut zg).expect("copy z");
        assert_close_slice(&yg, &a, 0.0, 0.0, "sqrtm_init_Y");
        assert_close_slice(&zg, &z_exp, 0.0, 0.0, "sqrtm_init_Z");
    }

    // --- iter: out=(M+I)/2 ---
    {
        let m: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-2.0, 2.0)).collect();
        let mut expect = vec![0.0f32; n * n];
        for col in 0..n {
            for row in 0..n {
                let d = if row == col { 1.0 } else { 0.0 };
                expect[row + col * n] = (m[row + col * n] + d) * 0.5;
            }
        }
        let (module, entry) = module_with_entry(&joined, "sqrtm_iter");
        let kernel = load_kernel(&module, &entry);
        let d_m = DeviceBuffer::<f32>::from_host(&m).expect("d_m");
        let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n]).expect("d_out");
        kernel
            .launch(
                &params,
                &stream,
                &(d_m.as_device_ptr(), d_out.as_device_ptr(), n as u32),
            )
            .expect("launch iter");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32; n * n];
        d_out.copy_to_host(&mut got).expect("copy");
        assert_close_slice(&got, &expect, 1e-6, 1e-6, "sqrtm_iter");
    }

    // --- conv: norm = sum (Y_new - Y_old)^2  (atomic accumulation) ---
    {
        let ynew: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
        let yold: Vec<f32> = (0..n * n).map(|_| rng.range_f32(-1.0, 1.0)).collect();
        let mut norm_exp = 0.0f64;
        for i in 0..n * n {
            let d = f64::from(ynew[i]) - f64::from(yold[i]);
            norm_exp += d * d;
        }
        let (module, entry) = module_with_entry(&joined, "sqrtm_conv");
        let kernel = load_kernel(&module, &entry);
        let d_yn = DeviceBuffer::<f32>::from_host(&ynew).expect("d_yn");
        let d_yo = DeviceBuffer::<f32>::from_host(&yold).expect("d_yo");
        let d_norm = DeviceBuffer::<f32>::from_host(&[0.0f32]).expect("d_norm");
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_yn.as_device_ptr(),
                    d_yo.as_device_ptr(),
                    d_norm.as_device_ptr(),
                    n as u32,
                ),
            )
            .expect("launch conv");
        stream.synchronize().expect("sync");
        let mut got = vec![0.0f32];
        d_norm.copy_to_host(&mut got).expect("copy");
        assert!(
            close(got[0], norm_exp as f32, 1e-4, 1e-4),
            "sqrtm_conv norm gpu={} cpu={}",
            got[0],
            norm_exp
        );
    }
}

// ===========================================================================
// Placeholder kernels (batched + QZ): JIT-load + launch validation only
// ===========================================================================

#[test]
fn placeholder_kernels_load_and_launch() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let stream = Stream::new(&fx.ctx).expect("stream");
    let sm = fx.sm;

    // batched_lu.
    {
        let n = 16usize;
        let batch = 4u32;
        let ptx = crate::dense::batched::emit_batched_lu::<f32>(sm, n).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("solver_batched_lu_f32_{n}"));
        let d_mat =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n * batch as usize]).expect("d_mat");
        let d_piv = DeviceBuffer::<u32>::from_host(&vec![0u32; n * batch as usize]).expect("d_piv");
        let params = LaunchParams::new(batch, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_mat.as_device_ptr(),
                    d_piv.as_device_ptr(),
                    n as u32,
                    batch,
                ),
            )
            .expect("launch batched_lu");
        stream.synchronize().expect("sync");
    }
    // batched_qr.
    {
        let m = 16usize;
        let n = 8usize;
        let batch = 3u32;
        let ptx = crate::dense::batched::emit_batched_qr::<f32>(sm, m, n).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("solver_batched_qr_f32_{m}x{n}"));
        let d_mat =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; m * n * batch as usize]).expect("d_mat");
        let d_tau =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * batch as usize]).expect("d_tau");
        let params = LaunchParams::new(batch, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_mat.as_device_ptr(),
                    d_tau.as_device_ptr(),
                    m as u32,
                    n as u32,
                    batch,
                ),
            )
            .expect("launch batched_qr");
        stream.synchronize().expect("sync");
    }
    // batched_cholesky.
    {
        let n = 16usize;
        let batch = 3u32;
        let ptx = crate::dense::batched::emit_batched_cholesky::<f32>(sm, n).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("solver_batched_cholesky_f32_{n}"));
        let d_mat =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n * batch as usize]).expect("d_mat");
        let params = LaunchParams::new(batch, 32u32);
        kernel
            .launch(&params, &stream, &(d_mat.as_device_ptr(), n as u32, batch))
            .expect("launch batched_cholesky");
        stream.synchronize().expect("sync");
    }
    // batched_solve.
    {
        let n = 16usize;
        let nrhs = 4usize;
        let batch = 3u32;
        let ptx = crate::dense::batched::emit_batched_solve::<f32>(sm, n, nrhs).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("solver_batched_solve_f32_{n}_{nrhs}"));
        let d_lu =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * n * batch as usize]).expect("d_lu");
        let d_b =
            DeviceBuffer::<f32>::from_host(&vec![0.0f32; n * nrhs * batch as usize]).expect("d_b");
        let d_piv = DeviceBuffer::<u32>::from_host(&vec![0u32; n * batch as usize]).expect("d_piv");
        let params = LaunchParams::new(batch, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_lu.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_piv.as_device_ptr(),
                    n as u32,
                    nrhs as u32,
                    batch,
                ),
            )
            .expect("launch batched_solve");
        stream.synchronize().expect("sync");
    }
    // QZ hessenberg reduction.
    {
        let n = 8usize;
        let ptx = crate::dense::qz::generate_hessenberg_reduction_ptx(n as u32, sm).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("qz_hessenberg_reduction_{n}"));
        let buf = |len: usize| DeviceBuffer::<f32>::from_host(&vec![0.0f32; len]).expect("buf");
        let a = buf(n * n);
        let bb = buf(n * n);
        let q = buf(n * n);
        let z = buf(n * n);
        let params = LaunchParams::new(1u32, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    a.as_device_ptr(),
                    bb.as_device_ptr(),
                    q.as_device_ptr(),
                    z.as_device_ptr(),
                    n as u32,
                ),
            )
            .expect("launch qz_hess");
        stream.synchronize().expect("sync");
    }
    // QZ sweep.
    {
        let n = 8usize;
        let ptx = crate::dense::qz::generate_qz_sweep_ptx(n as u32, sm).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("qz_sweep_{n}"));
        let buf = |len: usize| DeviceBuffer::<f32>::from_host(&vec![0.0f32; len]).expect("buf");
        let a = buf(n * n);
        let bb = buf(n * n);
        let q = buf(n * n);
        let z = buf(n * n);
        let params = LaunchParams::new(1u32, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    a.as_device_ptr(),
                    bb.as_device_ptr(),
                    q.as_device_ptr(),
                    z.as_device_ptr(),
                    0u32,
                    (n - 1) as u32,
                    n as u32,
                ),
            )
            .expect("launch qz_sweep");
        stream.synchronize().expect("sync");
    }
    // QZ eigenvalue extraction.
    {
        let n = 8usize;
        let ptx = crate::dense::qz::generate_eigenvalue_extraction_ptx(n as u32, sm).expect("ptx");
        let kernel = load_kernel(&ptx, &format!("qz_eigenvalue_extract_{n}"));
        let buf = |len: usize| DeviceBuffer::<f32>::from_host(&vec![0.0f32; len]).expect("buf");
        let s = buf(n * n);
        let t = buf(n * n);
        let ar = buf(n);
        let ai = buf(n);
        let beta = buf(n);
        let params = LaunchParams::new(1u32, 32u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    s.as_device_ptr(),
                    t.as_device_ptr(),
                    ar.as_device_ptr(),
                    ai.as_device_ptr(),
                    beta.as_device_ptr(),
                    n as u32,
                ),
            )
            .expect("launch qz_eig");
        stream.synchronize().expect("sync");
    }
}

// ===========================================================================
// Host-launcher regression: gemm_update grid mapping (X<-col, Y<-row)
// ===========================================================================

#[test]
fn launch_gemm_update_tall_block_full_coverage() {
    use crate::handle::SolverHandle;

    let Some(fx) = gpu_fixture() else {
        return;
    };
    let handle = SolverHandle::new(&fx.ctx).expect("handle");

    // A *tall* trailing block (rrows > one tile, rcols < one tile). With the
    // historical grid swap (grid_x sized by rrows, grid_y by rcols) the rows
    // beyond a single 16-row tile were never written, so this would mismatch.
    let rrows = 40usize;
    let rcols = 8usize;
    let kk = 5usize;
    let lda = rrows;
    let ldb = kk;
    let ldc = rrows;
    let mut rng = Lcg::new(0x6E33_0001);

    let a: Vec<f32> = (0..lda * kk).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let bmat: Vec<f32> = (0..ldb * rcols).map(|_| rng.range_f32(-1.0, 1.0)).collect();
    let c0: Vec<f32> = (0..ldc * rcols).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    let mut expect = c0.clone();
    for col in 0..rcols {
        for row in 0..rrows {
            let mut acc = 0.0f32;
            for k in 0..kk {
                acc = a[row + k * lda].mul_add(bmat[k + col * ldb], acc);
            }
            expect[row + col * ldc] -= acc;
        }
    }

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&bmat).expect("d_b");
    let d_c = DeviceBuffer::<f32>::from_host(&c0).expect("d_c");

    crate::dense::lu::launch_gemm_update::<f32>(
        &handle,
        d_a.as_device_ptr(),
        d_b.as_device_ptr(),
        d_c.as_device_ptr(),
        rrows as u32,
        rcols as u32,
        kk as u32,
        lda as u32,
        ldb as u32,
        ldc as u32,
    )
    .expect("launch_gemm_update");
    handle.stream().synchronize().expect("sync");

    let mut got = vec![0.0f32; ldc * rcols];
    d_c.copy_to_host(&mut got).expect("copy");
    assert_close_slice(&got, &expect, 1e-4, 1e-5, "launch_gemm_update_tall");
}
