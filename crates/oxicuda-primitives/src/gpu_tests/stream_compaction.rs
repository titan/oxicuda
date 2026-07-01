//! On-device validation of the stream-compaction kernels: histogram,
//! select-if, partition-if, select-unique, and run-length encode.
//!
//! The multi-kernel device algorithms (select / partition / unique / rle)
//! materialise a per-element flag (or run-head) array on the GPU, then need the
//! EXCLUSIVE prefix sum of those flags as scatter offsets. That intermediate
//! scan is computed here on the host (it is the oracle "glue"); the GPU flag and
//! gather/scatter kernels on either side are what these tests validate.

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use super::{Lcg, gpu_fixture, grid_1d, load_kernel, params};
use crate::device::histogram::{
    DeviceHistogramConfig, DeviceHistogramMode, DeviceHistogramTemplate,
};
use crate::device::partition::{
    DevicePartitionConfig, DevicePartitionTemplate, DeviceSelectUniqueConfig,
    DeviceSelectUniqueTemplate, reference_partition, reference_select_unique,
};
use crate::device::run_length_encode::{
    DeviceRunLengthEncodeConfig, DeviceRunLengthEncodeTemplate, reference_run_length_encode,
};
use crate::device::select::{DeviceSelectConfig, DeviceSelectTemplate, SelectPredicate};
use crate::host_reference::{
    reference_histogram_even, reference_histogram_modulo, reference_select,
};

/// Exclusive prefix sum of a u32 flag array into u64, plus the total.
fn exclusive_scan_u64(flags: &[u32]) -> (Vec<u64>, u64) {
    let mut out = Vec::with_capacity(flags.len());
    let mut acc = 0u64;
    for &f in flags {
        out.push(acc);
        acc += u64::from(f);
    }
    (out, acc)
}

// ===========================================================================
// histogram  (init zero + privatised count with atomic global merge)
// ===========================================================================

#[test]
fn histogram_modulo_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let num_bins = 16u32;
    let n = 4000usize;
    let mut rng = Lcg::new(0x_8157_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.below(1000)).collect();
    let expected = reference_histogram_modulo(&input, num_bins);

    let (init_ptx, count_ptx) = DeviceHistogramTemplate::new(DeviceHistogramConfig {
        ty: PtxType::U32,
        num_bins,
        block_size: block,
        mode: DeviceHistogramMode::Modulo,
    })
    .generate(fx.sm)
    .expect("gen histogram");
    let k_init = load_kernel(&init_ptx, "histogram_init_u32");
    let k_count = load_kernel(
        &count_ptx,
        &format!("histogram_count_modulo_u32_{num_bins}bins_bs{block}"),
    );
    let stream = fx.stream();

    let d_hist = DeviceBuffer::<u32>::from_host(&vec![0u32; num_bins as usize]).expect("d_hist");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");

    k_init
        .launch(
            &params(grid_1d(num_bins, block), block),
            &stream,
            &(d_hist.as_device_ptr(), num_bins),
        )
        .expect("launch init");
    k_count
        .launch(
            &params(grid_1d(n as u32, block), block),
            &stream,
            &(
                d_hist.as_device_ptr(),
                d_in.as_device_ptr(),
                n as u64,
                num_bins,
            ),
        )
        .expect("launch count");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; num_bins as usize];
    d_hist.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "histogram modulo");
    assert_eq!(got.iter().sum::<u32>(), n as u32, "all elements counted");
}

#[test]
fn histogram_even_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let num_bins = 16u32;
    let (lo, hi) = (0u32, 1024u32);
    let n = 4000usize;
    let mut rng = Lcg::new(0x_E7E2_0002);
    // Some values land outside [lo, hi) and must be dropped (bounds guard).
    let input: Vec<u32> = (0..n).map(|_| rng.below(1200)).collect();
    let expected = reference_histogram_even(&input, lo, hi, num_bins);

    let (init_ptx, count_ptx) = DeviceHistogramTemplate::new(DeviceHistogramConfig {
        ty: PtxType::U32,
        num_bins,
        block_size: block,
        mode: DeviceHistogramMode::EvenRange,
    })
    .generate(fx.sm)
    .expect("gen histogram");
    let k_init = load_kernel(&init_ptx, "histogram_init_u32");
    let k_count = load_kernel(
        &count_ptx,
        &format!("histogram_count_even_u32_{num_bins}bins_bs{block}"),
    );
    let stream = fx.stream();

    let d_hist = DeviceBuffer::<u32>::from_host(&vec![0u32; num_bins as usize]).expect("d_hist");
    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");

    k_init
        .launch(
            &params(grid_1d(num_bins, block), block),
            &stream,
            &(d_hist.as_device_ptr(), num_bins),
        )
        .expect("launch init");
    k_count
        .launch(
            &params(grid_1d(n as u32, block), block),
            &stream,
            &(
                d_hist.as_device_ptr(),
                d_in.as_device_ptr(),
                n as u64,
                num_bins,
                lo,
                hi,
            ),
        )
        .expect("launch count");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; num_bins as usize];
    d_hist.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "histogram even-range");
}

// ===========================================================================
// select-if  (flag + host exclusive scan + gather)
// ===========================================================================

#[test]
fn select_if_positive_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize;
    let mut rng = Lcg::new(0x_5E1E_0001);
    let input: Vec<i32> = (0..n)
        .map(|_| (rng.below(100) as i32) - 50) // [-50, 49]
        .collect();
    let expected = reference_select(&input, |x| x > 0);

    let (flag_ptx, gather_ptx) = DeviceSelectTemplate::new(DeviceSelectConfig {
        ty: PtxType::S32,
        pred: SelectPredicate::Positive,
        block_size: block,
    })
    .generate(fx.sm)
    .expect("gen select");
    let k_flag = load_kernel(
        &flag_ptx,
        &format!("device_select_flag_positive_s32_bs{block}"),
    );
    let k_gather = load_kernel(
        &gather_ptx,
        &format!("device_select_gather_positive_s32_bs{block}"),
    );
    let stream = fx.stream();

    let d_in = DeviceBuffer::<i32>::from_host(&input).expect("d_in");
    let d_flags = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_flags");
    let grid = grid_1d(n as u32, block);

    k_flag
        .launch(
            &params(grid, block),
            &stream,
            &(d_flags.as_device_ptr(), d_in.as_device_ptr(), n as u64),
        )
        .expect("launch flag");
    stream.synchronize().expect("sync");

    let mut flags = vec![0u32; n];
    d_flags.copy_to_host(&mut flags).expect("copy flags");
    let (offsets, count) = exclusive_scan_u64(&flags);
    assert_eq!(count as usize, expected.len(), "kept count");

    let d_off = DeviceBuffer::<u64>::from_host(&offsets).expect("d_off");
    let d_out = DeviceBuffer::<i32>::from_host(&vec![0i32; count as usize]).expect("d_out");

    k_gather
        .launch(
            &params(grid, block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_in.as_device_ptr(),
                d_flags.as_device_ptr(),
                d_off.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch gather");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; count as usize];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "select_if positive");
}

// ===========================================================================
// partition-if  (flag + host exclusive scan + two-output scatter)
// ===========================================================================

#[test]
fn partition_if_positive_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize;
    let mut rng = Lcg::new(0x_9A27_0001);
    let input: Vec<i32> = (0..n).map(|_| (rng.below(100) as i32) - 50).collect();
    let (exp_a, exp_b) = reference_partition(&input, |x| x > 0);

    let (flag_ptx, scatter_ptx) = DevicePartitionTemplate::new(
        DevicePartitionConfig::new(PtxType::S32, SelectPredicate::Positive, block).expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen partition");
    let k_flag = load_kernel(&flag_ptx, &format!("partition_flag_positive_s32_bs{block}"));
    let k_scatter = load_kernel(
        &scatter_ptx,
        &format!("partition_scatter_positive_s32_bs{block}"),
    );
    let stream = fx.stream();

    let d_in = DeviceBuffer::<i32>::from_host(&input).expect("d_in");
    let d_flags = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_flags");
    let grid = grid_1d(n as u32, block);

    k_flag
        .launch(
            &params(grid, block),
            &stream,
            &(d_flags.as_device_ptr(), d_in.as_device_ptr(), n as u64),
        )
        .expect("launch flag");
    stream.synchronize().expect("sync");

    let mut flags = vec![0u32; n];
    d_flags.copy_to_host(&mut flags).expect("copy flags");
    let (rank_a, count_a) = exclusive_scan_u64(&flags);
    assert_eq!(count_a as usize, exp_a.len(), "kept count");
    let count_b = n - count_a as usize;

    let d_rank = DeviceBuffer::<u64>::from_host(&rank_a).expect("d_rank");
    let d_a = DeviceBuffer::<i32>::from_host(&vec![0i32; count_a as usize]).expect("d_a");
    let d_b = DeviceBuffer::<i32>::from_host(&vec![0i32; count_b]).expect("d_b");

    k_scatter
        .launch(
            &params(grid, block),
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_in.as_device_ptr(),
                d_flags.as_device_ptr(),
                d_rank.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch scatter");
    stream.synchronize().expect("sync");

    let mut got_a = vec![0i32; count_a as usize];
    let mut got_b = vec![0i32; count_b];
    d_a.copy_to_host(&mut got_a).expect("copy a");
    d_b.copy_to_host(&mut got_b).expect("copy b");
    assert_eq!(got_a, exp_a, "partition kept");
    assert_eq!(got_b, exp_b, "partition rejected");
}

// ===========================================================================
// select-unique  (consecutive-unique compaction)
// ===========================================================================

#[test]
fn select_unique_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 2500usize;
    let mut rng = Lcg::new(0x_4E12_0001);
    // Small alphabet with repeats to create many consecutive runs.
    let input: Vec<u32> = (0..n).map(|_| rng.below(8)).collect();
    let expected = reference_select_unique(&input);

    let (head_ptx, gather_ptx) = DeviceSelectUniqueTemplate::new(
        DeviceSelectUniqueConfig::new(PtxType::U32, block).expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen unique");
    let k_head = load_kernel(&head_ptx, &format!("select_unique_head_u32_bs{block}"));
    let k_gather = load_kernel(&gather_ptx, &format!("select_unique_gather_u32_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_heads = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_heads");
    let grid = grid_1d(n as u32, block);

    k_head
        .launch(
            &params(grid, block),
            &stream,
            &(d_heads.as_device_ptr(), d_in.as_device_ptr(), n as u64),
        )
        .expect("launch head");
    stream.synchronize().expect("sync");

    let mut heads = vec![0u32; n];
    d_heads.copy_to_host(&mut heads).expect("copy heads");
    let (out_idx, count) = exclusive_scan_u64(&heads);
    assert_eq!(count as usize, expected.len(), "unique count");

    let d_idx = DeviceBuffer::<u64>::from_host(&out_idx).expect("d_idx");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0u32; count as usize]).expect("d_out");

    k_gather
        .launch(
            &params(grid, block),
            &stream,
            &(
                d_out.as_device_ptr(),
                d_in.as_device_ptr(),
                d_heads.as_device_ptr(),
                d_idx.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch gather");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; count as usize];
    d_out.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "select_unique");
}

// ===========================================================================
// run-length encode  (head + gather unique/starts + lengths)
// ===========================================================================

#[test]
fn run_length_encode_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 2500usize;
    let mut rng = Lcg::new(0x_27EE_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.below(6)).collect();
    let (exp_vals, exp_lens) = reference_run_length_encode(&input);

    let (head_ptx, gather_ptx, lengths_ptx) = DeviceRunLengthEncodeTemplate::new(
        DeviceRunLengthEncodeConfig::new(PtxType::U32, block).expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen rle");
    let k_head = load_kernel(&head_ptx, &format!("rle_head_u32_bs{block}"));
    let k_gather = load_kernel(&gather_ptx, &format!("rle_gather_u32_bs{block}"));
    let k_lengths = load_kernel(&lengths_ptx, &format!("rle_lengths_u32_bs{block}"));
    let stream = fx.stream();

    let d_in = DeviceBuffer::<u32>::from_host(&input).expect("d_in");
    let d_heads = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_heads");
    let grid = grid_1d(n as u32, block);

    k_head
        .launch(
            &params(grid, block),
            &stream,
            &(d_heads.as_device_ptr(), d_in.as_device_ptr(), n as u64),
        )
        .expect("launch head");
    stream.synchronize().expect("sync");

    let mut heads = vec![0u32; n];
    d_heads.copy_to_host(&mut heads).expect("copy heads");
    let (run_idx, num_runs_u64) = exclusive_scan_u64(&heads);
    let num_runs = num_runs_u64 as usize;
    assert_eq!(num_runs, exp_vals.len(), "run count");

    let d_ridx = DeviceBuffer::<u64>::from_host(&run_idx).expect("d_ridx");
    let d_uniq = DeviceBuffer::<u32>::from_host(&vec![0u32; num_runs]).expect("d_uniq");
    let d_starts = DeviceBuffer::<u64>::from_host(&vec![0u64; num_runs]).expect("d_starts");

    k_gather
        .launch(
            &params(grid, block),
            &stream,
            &(
                d_uniq.as_device_ptr(),
                d_starts.as_device_ptr(),
                d_in.as_device_ptr(),
                d_heads.as_device_ptr(),
                d_ridx.as_device_ptr(),
                n as u64,
            ),
        )
        .expect("launch gather");

    let d_lens = DeviceBuffer::<u64>::from_host(&vec![0u64; num_runs]).expect("d_lens");
    k_lengths
        .launch(
            &params(grid_1d(num_runs as u32, block), block),
            &stream,
            &(
                d_lens.as_device_ptr(),
                d_starts.as_device_ptr(),
                num_runs as u64,
                n as u64,
            ),
        )
        .expect("launch lengths");
    stream.synchronize().expect("sync");

    let mut got_vals = vec![0u32; num_runs];
    let mut got_lens = vec![0u64; num_runs];
    d_uniq.copy_to_host(&mut got_vals).expect("copy vals");
    d_lens.copy_to_host(&mut got_lens).expect("copy lens");
    assert_eq!(got_vals, exp_vals, "rle unique values");
    assert_eq!(got_lens, exp_lens, "rle run lengths");
}
