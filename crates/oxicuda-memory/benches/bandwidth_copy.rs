//! Memcpy bandwidth bench — verifies the **NF2** non-functional requirement
//! (H2D / D2H copy bandwidth ≥ 95 % of PCIe theoretical bandwidth).
//!
//! # What it measures
//!
//! Five criterion groups exercise the synchronous copy entry points across a
//! sweep of payload sizes from 4 KiB up to 256 MiB:
//!
//! | Group           | Direction          | Source / dest pairing       |
//! |-----------------|--------------------|-----------------------------|
//! | `h2d_pageable`  | host → device      | `Vec<f32>` → `DeviceBuffer` |
//! | `h2d_pinned`    | host → device      | `PinnedBuffer` → `DeviceBuffer` |
//! | `d2h_pageable`  | device → host      | `DeviceBuffer` → `Vec<f32>` |
//! | `d2h_pinned`    | device → host      | `DeviceBuffer` → `PinnedBuffer` |
//! | `d2d`           | device → device    | `DeviceBuffer` → `DeviceBuffer` |
//!
//! Each `bench_with_input` annotates `Throughput::Bytes(...)` so criterion
//! emits GiB/s automatically.  An auxiliary `report_nf2` helper logs the
//! measured peak vs. PCIe theoretical bandwidth (Gen3 / Gen4 / Gen5 ×16) so the
//! 95 % gate can be checked manually on Linux + NVIDIA.
//!
//! # Skip-on-no-GPU policy
//!
//! Every bench function constructs the CUDA context first; on macOS or any
//! host without a working driver, the construction fails, the function logs a
//! `skip:` notice on stderr and returns without invoking `b.iter`.  This keeps
//! the bench compile-clean on CI runners that do not have NVIDIA hardware.
//!
//! # PCIe generation override
//!
//! The reference peak used in `report_nf2` defaults to **PCIe Gen4 ×16**.
//! Set the environment variable `OXI_PCIE_GEN` to `3`, `4`, or `5` to override
//! the assumed generation when comparing the measurement to a theoretical max.

use std::hint::black_box;
use std::time::Instant;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxicuda_driver::{Context, Device, init};
use oxicuda_memory::copy::{copy_dtod, copy_dtoh, copy_htod};
use oxicuda_memory::{
    BandwidthMeasurement, BandwidthProfiler, DeviceBuffer, PinnedBuffer, TransferDirection,
    bandwidth_utilization, describe_bandwidth, format_bytes, theoretical_peak_bandwidth,
};

const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Default PCIe generation assumed when `OXI_PCIE_GEN` is unset.
const DEFAULT_PCIE_GEN: u32 = 4;

/// Default PCIe lane count (×16 — the standard slot for discrete GPUs).
const PCIE_LANES: u32 = 16;

/// Reads the assumed PCIe generation from the `OXI_PCIE_GEN` environment
/// variable, falling back to [`DEFAULT_PCIE_GEN`] (Gen4) on parse failure.
///
/// Only `3`, `4`, and `5` are accepted; other values revert to the default.
fn pcie_gen_from_env() -> u32 {
    match std::env::var("OXI_PCIE_GEN").ok().as_deref() {
        Some("3") => 3,
        Some("4") => 4,
        Some("5") => 5,
        _ => DEFAULT_PCIE_GEN,
    }
}

fn setup_context() -> Option<Context> {
    init().ok()?;
    if Device::count().ok()? <= 0 {
        return None;
    }
    let device = Device::get(0).ok()?;
    Context::new(&device).ok()
}

fn measure_h2d<T: Copy>(
    device: &mut DeviceBuffer<T>,
    host: &[T],
    warmup: u32,
    iters: u32,
) -> Option<BandwidthMeasurement> {
    for _ in 0..warmup {
        copy_htod(device, host).ok()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        copy_htod(device, host).ok()?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    Some(BandwidthMeasurement::new(
        TransferDirection::HostToDevice,
        std::mem::size_of_val(host),
        elapsed_ms,
    ))
}

fn measure_d2h<T: Copy>(
    host: &mut [T],
    device: &DeviceBuffer<T>,
    warmup: u32,
    iters: u32,
) -> Option<BandwidthMeasurement> {
    for _ in 0..warmup {
        copy_dtoh(host, device).ok()?;
    }

    let start = Instant::now();
    for _ in 0..iters {
        copy_dtoh(host, device).ok()?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(iters);
    Some(BandwidthMeasurement::new(
        TransferDirection::DeviceToHost,
        std::mem::size_of_val(host),
        elapsed_ms,
    ))
}

fn report_nf2(size_elements: usize) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("[bandwidth_copy] no CUDA device/context available, skipping NF2 report");
            return;
        }
    };

    let mut device = match DeviceBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] device allocation failed: {err}");
            return;
        }
    };

    let host_src_pageable = vec![1.0f32; size_elements];
    let mut host_dst_pageable = vec![0.0f32; size_elements];

    let mut host_src_pinned = match PinnedBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] pinned src allocation failed: {err}");
            return;
        }
    };
    host_src_pinned.as_mut_slice().fill(1.0);

    let mut host_dst_pinned = match PinnedBuffer::<f32>::alloc(size_elements) {
        Ok(buf) => buf,
        Err(err) => {
            eprintln!("[bandwidth_copy] pinned dst allocation failed: {err}");
            return;
        }
    };

    let mut profiler = BandwidthProfiler::with_iterations(3, 20);

    if let Some(m) = measure_h2d(
        &mut device,
        host_src_pageable.as_slice(),
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_d2h(
        host_dst_pageable.as_mut_slice(),
        &device,
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_h2d(
        &mut device,
        host_src_pinned.as_slice(),
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }
    if let Some(m) = measure_d2h(
        host_dst_pinned.as_mut_slice(),
        &device,
        profiler.warmup_iterations,
        profiler.benchmark_iterations,
    ) {
        profiler.record(m);
    }

    let assumed_gen = pcie_gen_from_env();
    let pcie3_x16_peak = theoretical_peak_bandwidth(3, PCIE_LANES);
    let pcie4_x16_peak = theoretical_peak_bandwidth(4, PCIE_LANES);
    let pcie5_x16_peak = theoretical_peak_bandwidth(5, PCIE_LANES);
    let assumed_peak = theoretical_peak_bandwidth(assumed_gen, PCIE_LANES);
    let summary = profiler.summary();

    eprintln!(
        "[bandwidth_copy] NF2 report for {} (assumed PCIe{}x{} = {:.2} GB/s; \
         reference Gen3x16 {:.2} GB/s, Gen4x16 {:.2} GB/s, Gen5x16 {:.2} GB/s)",
        format_bytes(size_elements * F32_BYTES),
        assumed_gen,
        PCIE_LANES,
        assumed_peak,
        pcie3_x16_peak,
        pcie4_x16_peak,
        pcie5_x16_peak,
    );

    for direction in [
        TransferDirection::HostToDevice,
        TransferDirection::DeviceToHost,
    ] {
        if let Some(dir) = summary
            .per_direction
            .iter()
            .filter(|d| d.direction == direction)
            .max_by(|a, b| a.max_bandwidth_gbps.total_cmp(&b.max_bandwidth_gbps))
        {
            let util_assumed = bandwidth_utilization(dir.max_bandwidth_gbps, assumed_peak) * 100.0;
            let util_pcie4 = bandwidth_utilization(dir.max_bandwidth_gbps, pcie4_x16_peak) * 100.0;
            let util_pcie3 = bandwidth_utilization(dir.max_bandwidth_gbps, pcie3_x16_peak) * 100.0;
            let util_pcie5 = bandwidth_utilization(dir.max_bandwidth_gbps, pcie5_x16_peak) * 100.0;
            let nf2_target_pct = 95.0_f64;
            let nf2_status = if util_assumed >= nf2_target_pct {
                "PASS"
            } else {
                "BELOW-TARGET"
            };
            eprintln!(
                "  {} best: {} (PCIe{}x{} {:.1}% [{} {:.0}% gate]; \
                 Gen3x16 {:.1}%, Gen4x16 {:.1}%, Gen5x16 {:.1}%)",
                direction,
                describe_bandwidth(dir.max_bandwidth_gbps),
                assumed_gen,
                PCIE_LANES,
                util_assumed,
                nf2_status,
                nf2_target_pct,
                util_pcie3,
                util_pcie4,
                util_pcie5,
            );
        }
    }
}

/// Size sweep used by every bench group: 4 KiB → 256 MiB in powers of four
/// (with a few intermediate steps), expressed as `(label, byte_size)`.
///
/// `byte_size` is the total payload moved per copy; element counts are derived
/// per-bench by dividing by `size_of::<T>()`.
const SIZE_SWEEP: &[(&str, usize)] = &[
    ("4KiB", 4 << 10),
    ("16KiB", 16 << 10),
    ("64KiB", 64 << 10),
    ("256KiB", 256 << 10),
    ("1MiB", 1 << 20),
    ("4MiB", 4 << 20),
    ("16MiB", 16 << 20),
    ("64MiB", 64 << 20),
    ("256MiB", 256 << 20),
];

/// Returns the element count (`f32`) for a given byte payload, ensuring it is
/// at least one element to satisfy [`DeviceBuffer::alloc`]'s non-zero
/// requirement.
fn elements_for_bytes(bytes: usize) -> usize {
    (bytes / F32_BYTES).max(1)
}

/// H2D from a pageable `Vec<f32>`.
fn bench_h2d_pageable(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("skip: no GPU (h2d_pageable)");
            return;
        }
    };

    // Emit the NF2 report once per run, anchored at 64 MiB (same as the prior
    // baseline) — it is independent of the size sweep below.
    report_nf2(elements_for_bytes(64 << 20));

    let mut group = c.benchmark_group("h2d_pageable");
    group.sample_size(20);

    for &(label, bytes) in SIZE_SWEEP {
        let elements = elements_for_bytes(bytes);
        let mut device = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[h2d_pageable] alloc {label}: {err}, skipping");
                continue;
            }
        };
        let host_src = vec![1.0_f32; elements];

        group.throughput(Throughput::Bytes((elements * F32_BYTES) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &elements, |b, _| {
            b.iter(|| {
                copy_htod(&mut device, black_box(host_src.as_slice())).ok();
            });
        });
    }

    group.finish();
}

/// H2D from a page-locked [`PinnedBuffer`].
fn bench_h2d_pinned(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("skip: no GPU (h2d_pinned)");
            return;
        }
    };

    let mut group = c.benchmark_group("h2d_pinned");
    group.sample_size(20);

    for &(label, bytes) in SIZE_SWEEP {
        let elements = elements_for_bytes(bytes);
        let mut device = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[h2d_pinned] device alloc {label}: {err}, skipping");
                continue;
            }
        };
        let mut host_src = match PinnedBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[h2d_pinned] pinned alloc {label}: {err}, skipping");
                continue;
            }
        };
        host_src.as_mut_slice().fill(1.0);

        group.throughput(Throughput::Bytes((elements * F32_BYTES) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &elements, |b, _| {
            b.iter(|| {
                copy_htod(&mut device, black_box(host_src.as_slice())).ok();
            });
        });
    }

    group.finish();
}

/// D2H into a pageable `Vec<f32>`.
fn bench_d2h_pageable(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("skip: no GPU (d2h_pageable)");
            return;
        }
    };

    let mut group = c.benchmark_group("d2h_pageable");
    group.sample_size(20);

    for &(label, bytes) in SIZE_SWEEP {
        let elements = elements_for_bytes(bytes);
        let device = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[d2h_pageable] alloc {label}: {err}, skipping");
                continue;
            }
        };
        let mut host_dst = vec![0.0_f32; elements];

        group.throughput(Throughput::Bytes((elements * F32_BYTES) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &elements, |b, _| {
            b.iter(|| {
                copy_dtoh(black_box(host_dst.as_mut_slice()), &device).ok();
            });
        });
    }

    group.finish();
}

/// D2H into a page-locked [`PinnedBuffer`].
fn bench_d2h_pinned(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("skip: no GPU (d2h_pinned)");
            return;
        }
    };

    let mut group = c.benchmark_group("d2h_pinned");
    group.sample_size(20);

    for &(label, bytes) in SIZE_SWEEP {
        let elements = elements_for_bytes(bytes);
        let device = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[d2h_pinned] device alloc {label}: {err}, skipping");
                continue;
            }
        };
        let mut host_dst = match PinnedBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[d2h_pinned] pinned alloc {label}: {err}, skipping");
                continue;
            }
        };

        group.throughput(Throughput::Bytes((elements * F32_BYTES) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &elements, |b, _| {
            b.iter(|| {
                copy_dtoh(black_box(host_dst.as_mut_slice()), &device).ok();
            });
        });
    }

    group.finish();
}

/// D2D between two `DeviceBuffer<f32>`s.
///
/// The 256 MiB step requires ~512 MiB of device memory (two buffers); on
/// constrained devices the inner allocation simply fails and the size is
/// skipped — the bench does not abort.
fn bench_d2d(c: &mut Criterion) {
    let _ctx = match setup_context() {
        Some(ctx) => ctx,
        None => {
            eprintln!("skip: no GPU (d2d)");
            return;
        }
    };

    let mut group = c.benchmark_group("d2d");
    group.sample_size(20);

    for &(label, bytes) in SIZE_SWEEP {
        let elements = elements_for_bytes(bytes);
        let src = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[d2d] src alloc {label}: {err}, skipping");
                continue;
            }
        };
        let mut dst = match DeviceBuffer::<f32>::alloc(elements) {
            Ok(buf) => buf,
            Err(err) => {
                eprintln!("[d2d] dst alloc {label}: {err}, skipping");
                continue;
            }
        };

        group.throughput(Throughput::Bytes((elements * F32_BYTES) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &elements, |b, _| {
            b.iter(|| {
                copy_dtod(&mut dst, black_box(&src)).ok();
            });
        });
    }

    group.finish();
}

criterion_group!(
    memory_copy_benches,
    bench_h2d_pageable,
    bench_h2d_pinned,
    bench_d2h_pageable,
    bench_d2h_pinned,
    bench_d2d,
);
criterion_main!(memory_copy_benches);
