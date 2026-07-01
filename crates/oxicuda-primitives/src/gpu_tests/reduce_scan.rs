//! On-device validation of the warp / block / device reduce & scan kernels,
//! plus segmented reduce/scan and decoupled-lookback scan.

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use super::{Lcg, gpu_fixture, grid_1d, load_kernel, params};
use crate::block::reduce::{BlockReduceConfig, BlockReduceTemplate};
use crate::block::scan::{BlockScanConfig, BlockScanTemplate};
use crate::device::decoupled_scan::{
    DecoupledScanConfig, DecoupledScanTemplate, ScanKind as DScanKind, reference_decoupled_scan_u64,
};
use crate::device::reduce::{DeviceReduceConfig, DeviceReduceTemplate};
use crate::device::scan::{DeviceScanConfig, DeviceScanTemplate};
use crate::device::segmented::{
    SegScanKind, SegmentedReduceConfig, SegmentedReduceTemplate, SegmentedScanConfig,
    SegmentedScanTemplate, reference_segmented_reduce_u64, reference_segmented_scan_u64,
};
use crate::ptx_helpers::ReduceOp;
use crate::warp::reduce::{WarpReduceConfig, WarpReduceTemplate};
use crate::warp::scan::{ScanKind as WScanKind, WarpScanConfig, WarpScanTemplate};

// ===========================================================================
// warp reduce  (single warp, shfl.bfly butterfly)
// ===========================================================================

fn run_warp_reduce_sum(n: usize) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x00FF_AA00 ^ n as u64);
    let input: Vec<u32> = (0..n).map(|_| rng.below(1000)).collect();
    let expected: u32 = input.iter().copied().sum();

    let ptx = WarpReduceTemplate::new(WarpReduceConfig {
        op: ReduceOp::Sum,
        ty: PtxType::U32,
        broadcast: false,
    })
    .generate(fx.sm)
    .expect("gen warp reduce");
    let kernel = load_kernel(&ptx, "warp_reduce_sum_u32");
    let stream = fx.stream();

    let d_out = DeviceBuffer::<u32>::from_host(&[0u32]).expect("d_out");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr(), n as u32),
        )
        .expect("launch warp_reduce");
    stream.synchronize().expect("sync");

    let mut got = [0u32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got[0], expected, "warp_reduce_sum n={n}");
}

#[test]
fn warp_reduce_sum_full_warp() {
    run_warp_reduce_sum(32);
}

#[test]
fn warp_reduce_sum_ragged() {
    run_warp_reduce_sum(19);
}

// ===========================================================================
// warp scan  (single warp, Hillis-Steele shfl.up)
// ===========================================================================

fn run_warp_scan(kind: WScanKind, n: usize) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x5CA4 ^ n as u64);
    let input: Vec<u32> = (0..n).map(|_| rng.below(100)).collect();

    let mut expected = vec![0u32; n];
    let mut acc = 0u32;
    for i in 0..n {
        match kind {
            WScanKind::Inclusive => {
                acc += input[i];
                expected[i] = acc;
            }
            WScanKind::Exclusive => {
                expected[i] = acc;
                acc += input[i];
            }
        }
    }

    let name = match kind {
        WScanKind::Inclusive => "warp_scan_sum_u32_inclusive",
        WScanKind::Exclusive => "warp_scan_sum_u32_exclusive",
    };
    let ptx = WarpScanTemplate::new(WarpScanConfig {
        op: ReduceOp::Sum,
        ty: PtxType::U32,
        kind,
    })
    .generate(fx.sm)
    .expect("gen warp scan");
    let kernel = load_kernel(&ptx, name);
    let stream = fx.stream();

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_out");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    kernel
        .launch(
            &params(1, 32),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr(), n as u32),
        )
        .expect("launch warp_scan");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "warp_scan {name} n={n}");
}

#[test]
fn warp_scan_inclusive_full() {
    run_warp_scan(WScanKind::Inclusive, 32);
}

#[test]
fn warp_scan_inclusive_ragged() {
    run_warp_scan(WScanKind::Inclusive, 21);
}

#[test]
fn warp_scan_exclusive_full() {
    run_warp_scan(WScanKind::Exclusive, 32);
}

// ===========================================================================
// block reduce  (warp partials in shared memory, warp-0 final reduce)
// ===========================================================================

fn run_block_reduce_sum(block: u32, n: usize) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xB10C ^ n as u64);
    let input: Vec<u32> = (0..n).map(|_| rng.below(500)).collect();
    let expected: u32 = input.iter().copied().sum();

    let ptx = BlockReduceTemplate::new(BlockReduceConfig {
        op: ReduceOp::Sum,
        ty: PtxType::U32,
        block_size: block,
        broadcast: false,
    })
    .generate(fx.sm)
    .expect("gen block reduce");
    let kernel = load_kernel(&ptx, &format!("block_reduce_sum_u32_bs{block}"));
    let stream = fx.stream();

    let d_out = DeviceBuffer::<u32>::from_host(&[0u32]).expect("d_out");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    kernel
        .launch(
            &params(1, block),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr(), n as u32),
        )
        .expect("launch block_reduce");
    stream.synchronize().expect("sync");

    let mut got = [0u32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got[0], expected, "block_reduce_sum bs={block} n={n}");
}

#[test]
fn block_reduce_sum_full_block() {
    run_block_reduce_sum(256, 256);
}

#[test]
fn block_reduce_sum_ragged() {
    run_block_reduce_sum(256, 200);
}

// ===========================================================================
// block scan  (Blelloch work-efficient scan)
// ===========================================================================

fn run_block_scan(kind: WScanKind, block: u32, n: usize) {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0xB5CA ^ n as u64);
    let input: Vec<u32> = (0..n).map(|_| rng.below(50)).collect();

    let mut expected = vec![0u32; n];
    let mut acc = 0u32;
    for i in 0..n {
        match kind {
            WScanKind::Inclusive => {
                acc += input[i];
                expected[i] = acc;
            }
            WScanKind::Exclusive => {
                expected[i] = acc;
                acc += input[i];
            }
        }
    }

    let suffix = match kind {
        WScanKind::Inclusive => "inclusive",
        WScanKind::Exclusive => "exclusive",
    };
    let ptx = BlockScanTemplate::new(BlockScanConfig {
        op: ReduceOp::Sum,
        ty: PtxType::U32,
        block_size: block,
        kind,
    })
    .generate(fx.sm)
    .expect("gen block scan");
    let kernel = load_kernel(&ptx, &format!("block_scan_sum_u32_bs{block}_{suffix}"));
    let stream = fx.stream();

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_out");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    kernel
        .launch(
            &params(1, block),
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr(), n as u32),
        )
        .expect("launch block_scan");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "block_scan_{suffix} bs={block} n={n}");
}

#[test]
fn block_scan_inclusive_full() {
    run_block_scan(WScanKind::Inclusive, 256, 256);
}

#[test]
fn block_scan_inclusive_ragged() {
    run_block_scan(WScanKind::Inclusive, 256, 173);
}

#[test]
fn block_scan_exclusive_full() {
    run_block_scan(WScanKind::Exclusive, 256, 256);
}

// ===========================================================================
// device reduce  (pass1 per-block partials -> pass2 final reduce)
// ===========================================================================

#[test]
fn device_reduce_sum_multiblock() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 5000usize; // 20 blocks
    let num_blocks = grid_1d(n as u32, block);

    let mut rng = Lcg::new(0xDE_D0_CE);
    let input: Vec<u32> = (0..n).map(|_| rng.below(100)).collect();
    let expected: u32 = input.iter().copied().sum();

    let (p1, p2) = DeviceReduceTemplate::new(DeviceReduceConfig::new(ReduceOp::Sum, PtxType::U32))
        .generate(fx.sm)
        .expect("gen device reduce");
    let k1 = load_kernel(&p1, &format!("device_reduce_pass1_sum_u32_bs{block}"));
    let k2 = load_kernel(&p2, &format!("device_reduce_pass2_sum_u32_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_partials =
        DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("d_part");
    let d_out = DeviceBuffer::<u32>::from_host(&[0u32]).expect("d_out");

    k1.launch(
        &params(num_blocks, block),
        &stream,
        &(d_partials.as_device_ptr(), d_in.as_device_ptr(), n as u64),
    )
    .expect("launch pass1");
    k2.launch(
        &params(1, block),
        &stream,
        &(
            d_out.as_device_ptr(),
            d_partials.as_device_ptr(),
            num_blocks,
        ),
    )
    .expect("launch pass2");
    stream.synchronize().expect("sync");

    let mut got = [0u32];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got[0], expected, "device_reduce_sum n={n}");
}

// ===========================================================================
// device scan  (block scan + block-sums aggregate + propagate)
// ===========================================================================

#[test]
fn device_scan_inclusive_multiblock() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize; // 12 blocks
    let num_blocks = grid_1d(n as u32, block);

    let mut rng = Lcg::new(0xD5CA_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.below(20)).collect();
    let mut expected = vec![0u32; n];
    let mut acc = 0u32;
    for i in 0..n {
        acc += input[i];
        expected[i] = acc;
    }

    let (blk, agg, prop) = DeviceScanTemplate::new(DeviceScanConfig {
        op: ReduceOp::Sum,
        ty: PtxType::U32,
        block_size: block,
        kind: WScanKind::Inclusive,
    })
    .generate(fx.sm)
    .expect("gen device scan");
    let k_blk = load_kernel(
        &blk,
        &format!("device_scan_block_sum_u32_inclusive_bs{block}"),
    );
    let k_agg = load_kernel(&agg, "device_scan_aggregate_sum_u32");
    let k_prop = load_kernel(
        &prop,
        &format!("device_scan_propagate_sum_u32_inclusive_bs{block}"),
    );
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_out");
    let d_sums = DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("d_sums");

    k_blk
        .launch(
            &params(num_blocks, block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_sums.as_device_ptr(),
                d_in.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch block");
    k_agg
        .launch(
            &params(1, block),
            &stream,
            &(d_sums.as_device_ptr(), num_blocks),
        )
        .expect("launch aggregate");
    k_prop
        .launch(
            &params(num_blocks, block),
            &stream,
            &(d_out.as_device_ptr(), d_sums.as_device_ptr(), n as u64),
        )
        .expect("launch propagate");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "device_scan_inclusive n={n}");
}

// ===========================================================================
// segmented reduce / scan  (one thread per segment)
// ===========================================================================

fn make_segments(rng: &mut Lcg, num_segments: usize, max_seg_len: u32) -> (Vec<u64>, Vec<u64>) {
    let mut offsets = vec![0u64];
    let mut total = 0u64;
    for _ in 0..num_segments {
        let len = u64::from(rng.below(max_seg_len) + 1);
        total += len;
        offsets.push(total);
    }
    let data: Vec<u64> = (0..total).map(|_| u64::from(rng.below(1000))).collect();
    (data, offsets)
}

#[test]
fn segmented_reduce_sum_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let num_segments = 40usize;
    let mut rng = Lcg::new(0x5E6_0001);
    let (data, offsets) = make_segments(&mut rng, num_segments, 12);
    let expected = reference_segmented_reduce_u64(ReduceOp::Sum, &data, &offsets);

    let ptx = SegmentedReduceTemplate::new(
        SegmentedReduceConfig::new(ReduceOp::Sum, PtxType::U64, block).expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen seg reduce");
    let kernel = load_kernel(&ptx, &format!("seg_reduce_sum_u64_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u64>::from_host(&data).expect("d_in");
    let d_off = DeviceBuffer::<u64>::from_host(&offsets).expect("d_off");
    let d_out = DeviceBuffer::<u64>::from_host(&vec![0u64; num_segments]).expect("d_out");

    // seg_reduce maps one BLOCK per segment (blockIdx.x == segment id).
    kernel
        .launch(
            &params(num_segments as u32, block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_in.as_device_ptr(),
                d_off.as_device_ptr(),
                num_segments as u64,
            ),
        )
        .expect("launch seg reduce");
    stream.synchronize().expect("sync");

    let mut got = vec![0u64; num_segments];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "segmented_reduce_sum");
}

#[test]
fn segmented_scan_inclusive_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let num_segments = 30usize;
    let mut rng = Lcg::new(0x5E6_5CA);
    let (data, offsets) = make_segments(&mut rng, num_segments, 10);
    let expected =
        reference_segmented_scan_u64(ReduceOp::Sum, SegScanKind::Inclusive, &data, &offsets);

    let ptx = SegmentedScanTemplate::new(
        SegmentedScanConfig::new(ReduceOp::Sum, PtxType::U64, SegScanKind::Inclusive, block)
            .expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen seg scan");
    let kernel = load_kernel(&ptx, &format!("seg_scan_inc_sum_u64_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u64>::from_host(&data).expect("d_in");
    let d_off = DeviceBuffer::<u64>::from_host(&offsets).expect("d_off");
    let d_out = DeviceBuffer::<u64>::from_host(&vec![0u64; data.len()]).expect("d_out");

    kernel
        .launch(
            &params(grid_1d(num_segments as u32, block), block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_in.as_device_ptr(),
                d_off.as_device_ptr(),
                num_segments as u64,
            ),
        )
        .expect("launch seg scan");
    stream.synchronize().expect("sync");

    let mut got = vec![0u64; data.len()];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "segmented_scan_inclusive");
}

// ===========================================================================
// decoupled-lookback scan  (single kernel, global status flags)
// ===========================================================================

#[test]
fn decoupled_scan_inclusive_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 4000usize;
    let num_blocks = grid_1d(n as u32, block);

    let mut rng = Lcg::new(0xDEC0_0501);
    let input: Vec<u32> = (0..n).map(|_| rng.below(20)).collect();
    let data64: Vec<u64> = input.iter().map(|&v| u64::from(v)).collect();
    let expected_u64 =
        reference_decoupled_scan_u64(ReduceOp::Sum, DScanKind::Inclusive, &data64, block);
    let expected: Vec<u32> = expected_u64.iter().map(|&v| v as u32).collect();

    let ptx = DecoupledScanTemplate::new(
        DecoupledScanConfig::with_kind(ReduceOp::Sum, PtxType::U32, block, DScanKind::Inclusive)
            .expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen decoupled");
    let kernel = load_kernel(&ptx, &format!("decoupled_scan_inc_sum_u32_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_out");
    // Per-block decoupled state: status flags (u32), aggregates (u32), prefixes (u32).
    let d_status = DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("d_st");
    let d_agg = DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("d_agg");
    let d_prefix = DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("d_pre");

    kernel
        .launch(
            &params(num_blocks, block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_in.as_device_ptr(),
                d_status.as_device_ptr(),
                d_agg.as_device_ptr(),
                d_prefix.as_device_ptr(),
                n as u64,
                num_blocks,
            ),
        )
        .expect("launch decoupled");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; n];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "decoupled_scan_inclusive n={n}");
}
