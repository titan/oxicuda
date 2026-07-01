//! On-device validation of the sort kernels: float-key twiddle bijection,
//! key+value merge, full merge sort, and per-segment bitonic sort.

use std::collections::BTreeSet;

use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use super::{Lcg, gpu_fixture, grid_1d, load_kernel, params};
use crate::sort::merge_pairs::{MergePairsConfig, MergePairsTemplate, reference_merge_pairs};
use crate::sort::merge_sort::{MergeSortConfig, MergeSortTemplate};
use crate::sort::onesweep::{OnesweepConfig, OnesweepTemplate, reference_onesweep_sort_u32};
use crate::sort::radix_sort::{RadixSortConfig, RadixSortTemplate};
use crate::sort::radix_sort_8bit::{
    RadixSort8Config, RadixSort8Template, reference_radix8_sort_u32,
};
use crate::sort::radix_sort_pairs::{
    FloatTwiddleConfig, FloatTwiddleTemplate, RadixPairsConfig, RadixPairsTemplate, SortOrder,
    reference_sort_pairs_by_key, twiddle_f32_forward, twiddle_f32_inverse,
};
use crate::sort::segmented_sort::{SegmentedSortConfig, SegmentedSortTemplate};
use oxicuda_launch::Kernel;

// ===========================================================================
// float-key twiddle  (radix-sortable bijection on f32 bits, in place on b32)
// ===========================================================================

#[test]
fn float_twiddle_forward_and_roundtrip() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 500usize;
    let mut rng = Lcg::new(0x7_DD1E);
    let values: Vec<f32> = (0..n).map(|_| rng.f32_in(-100.0, 100.0)).collect();
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    let fwd_expected: Vec<u32> = values.iter().map(|&v| twiddle_f32_forward(v)).collect();

    let (fwd_ptx, inv_ptx) =
        FloatTwiddleTemplate::new(FloatTwiddleConfig::new(PtxType::F32, block).expect("cfg"))
            .generate(fx.sm)
            .expect("gen twiddle");
    let k_fwd = load_kernel(&fwd_ptx, &format!("radix_float_twiddle_fwd_f32_bs{block}"));
    let k_inv = load_kernel(&inv_ptx, &format!("radix_float_twiddle_inv_f32_bs{block}"));
    let stream = fx.stream();

    let d_data = DeviceBuffer::<u32>::from_host(&bits).expect("d_data");

    // Forward: bits -> radix-sortable key.
    k_fwd
        .launch(
            &params(grid_1d(n as u32, block), block),
            &stream,
            &(d_data.as_device_ptr(), n as u64),
        )
        .expect("launch fwd");
    stream.synchronize().expect("sync");
    let mut fwd_got = vec![0u32; n];
    d_data.copy_to_host(&mut fwd_got).expect("copy fwd");
    assert_eq!(fwd_got, fwd_expected, "twiddle forward");

    // The twiddle must be monotone: sorting the keys ascending sorts the floats.
    let mut paired: Vec<(u32, f32)> = fwd_got
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    paired.sort_by_key(|&(k, _)| k);
    for w in paired.windows(2) {
        assert!(
            w[0].1 <= w[1].1,
            "twiddle not monotone: {} (key {}) before {} (key {})",
            w[0].1,
            w[0].0,
            w[1].1,
            w[1].0
        );
    }

    // Inverse must round-trip back to the original bits.
    k_inv
        .launch(
            &params(grid_1d(n as u32, block), block),
            &stream,
            &(d_data.as_device_ptr(), n as u64),
        )
        .expect("launch inv");
    stream.synchronize().expect("sync");
    let mut inv_got = vec![0u32; n];
    d_data.copy_to_host(&mut inv_got).expect("copy inv");
    assert_eq!(inv_got, bits, "twiddle inverse round-trip");

    // Cross-check the host inverse, too.
    for (i, &f) in fwd_expected.iter().enumerate() {
        assert_eq!(
            twiddle_f32_inverse(f).to_bits(),
            bits[i],
            "host inverse [{i}]"
        );
    }
}

// ===========================================================================
// merge pairs  (single key+value merge pass via co-rank binary search)
// ===========================================================================

#[test]
fn merge_pairs_single_pass_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let half = 300usize;
    let n = 2 * half;

    // Distinct keys so the merge tie-break is unambiguous.
    let mut rng = Lcg::new(0x_E26E);
    let mut set: BTreeSet<u32> = BTreeSet::new();
    while set.len() < n {
        set.insert(rng.below(1_000_000));
    }
    let mut all: Vec<u32> = set.into_iter().collect();
    // Shuffle deterministically, then split into two independently-sorted runs.
    for i in (1..all.len()).rev() {
        let j = rng.below((i + 1) as u32) as usize;
        all.swap(i, j);
    }
    let mut left_keys: Vec<u32> = all[..half].to_vec();
    let mut right_keys: Vec<u32> = all[half..].to_vec();
    left_keys.sort_unstable();
    right_keys.sort_unstable();
    // Values tag each key so we can confirm the value travels with its key.
    let left_vals: Vec<u32> = left_keys.iter().map(|&k| k ^ 0xA5A5_A5A5).collect();
    let right_vals: Vec<u32> = right_keys.iter().map(|&k| k ^ 0xA5A5_A5A5).collect();

    let lk64: Vec<u64> = left_keys.iter().map(|&k| u64::from(k)).collect();
    let rk64: Vec<u64> = right_keys.iter().map(|&k| u64::from(k)).collect();
    let lv64: Vec<u64> = left_vals.iter().map(|&v| u64::from(v)).collect();
    let rv64: Vec<u64> = right_vals.iter().map(|&v| u64::from(v)).collect();
    let (exp_keys, exp_vals) = reference_merge_pairs(&lk64, &lv64, &rk64, &rv64);
    let exp_keys: Vec<u32> = exp_keys.iter().map(|&k| k as u32).collect();
    let exp_vals: Vec<u32> = exp_vals.iter().map(|&v| v as u32).collect();

    let keys_in: Vec<u32> = left_keys.iter().chain(right_keys.iter()).copied().collect();
    let vals_in: Vec<u32> = left_vals.iter().chain(right_vals.iter()).copied().collect();

    let ptx = MergePairsTemplate::new(
        MergePairsConfig::new(PtxType::U32, PtxType::U32, block).expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen merge pairs");
    let kernel = load_kernel(&ptx, &format!("merge_pairs_u32_u32_bs{block}"));
    let stream = fx.stream();

    let d_ko = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_ko");
    let d_vo = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_vo");
    let d_ki = DeviceBuffer::<u32>::from_host(&keys_in).expect("d_ki");
    let d_vi = DeviceBuffer::<u32>::from_host(&vals_in).expect("d_vi");

    kernel
        .launch(
            &params(grid_1d(n as u32, block), block),
            &stream,
            &(
                d_ko.as_device_ptr(),
                d_vo.as_device_ptr(),
                d_ki.as_device_ptr(),
                d_vi.as_device_ptr(),
                n as u64,
                half as u64,
            ),
        )
        .expect("launch merge pairs");
    stream.synchronize().expect("sync");

    let mut got_keys = vec![0u32; n];
    let mut got_vals = vec![0u32; n];
    d_ko.copy_to_host(&mut got_keys).expect("copy keys");
    d_vo.copy_to_host(&mut got_vals).expect("copy vals");
    assert_eq!(got_keys, exp_keys, "merge_pairs keys");
    assert_eq!(got_vals, exp_vals, "merge_pairs vals");
}

// ===========================================================================
// merge sort  (block bitonic sort + ping-pong merge passes)
// ===========================================================================

#[test]
fn merge_sort_full_sorts_ascending() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 2000usize;
    let mut rng = Lcg::new(0x_5027_1234);
    let input: Vec<u32> = (0..n).map(|_| rng.below(100_000)).collect();
    let mut expected = input.clone();
    expected.sort_unstable();

    let (sort_ptx, merge_ptx) = MergeSortTemplate::new(MergeSortConfig {
        ty: PtxType::U32,
        block_size: block,
    })
    .generate(fx.sm)
    .expect("gen merge sort");
    let k_sort = load_kernel(&sort_ptx, &format!("merge_sort_blocks_u32_bs{block}"));
    let k_merge = load_kernel(&merge_ptx, &format!("merge_sort_merge_u32_bs{block}"));
    let stream = fx.stream();

    let d_a = DeviceBuffer::<u32>::from_host(&input).expect("d_a");
    let d_b = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_b");
    let grid = grid_1d(n as u32, block);

    // Phase 1: sort each block of `block` elements in place.
    k_sort
        .launch(
            &params(grid, block),
            &stream,
            &(d_a.as_device_ptr(), n as u64),
        )
        .expect("launch sort blocks");

    // Phase 2: ping-pong merge passes, doubling the run length each time.
    let mut merge_len = u64::from(block);
    let mut src_is_a = true;
    while merge_len < n as u64 {
        let (src, dst) = if src_is_a { (&d_a, &d_b) } else { (&d_b, &d_a) };
        k_merge
            .launch(
                &params(grid, block),
                &stream,
                &(
                    dst.as_device_ptr(),
                    src.as_device_ptr(),
                    n as u64,
                    merge_len,
                ),
            )
            .expect("launch merge");
        src_is_a = !src_is_a;
        merge_len *= 2;
    }
    stream.synchronize().expect("sync");

    let final_buf = if src_is_a { &d_a } else { &d_b };
    let mut got = vec![0u32; n];
    final_buf.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "merge_sort ascending");
}

// ===========================================================================
// segmented sort  (one block per segment, bitonic, in place)
// ===========================================================================

#[test]
fn segmented_sort_sorts_each_segment() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let num_segments = 24usize;
    let mut rng = Lcg::new(0x_5E65_0271);

    // Segment lengths in [1, block]; data is random per element.
    let mut offsets = vec![0u64];
    let mut total = 0u64;
    for _ in 0..num_segments {
        let len = u64::from(rng.below(block) + 1); // 1..=block
        total += len;
        offsets.push(total);
    }
    let data: Vec<u32> = (0..total).map(|_| rng.below(50_000)).collect();

    let mut expected = data.clone();
    for w in offsets.windows(2) {
        let (b, e) = (w[0] as usize, w[1] as usize);
        expected[b..e].sort_unstable();
    }

    let ptx =
        SegmentedSortTemplate::new(SegmentedSortConfig::new(PtxType::U32, block).expect("cfg"))
            .generate(fx.sm)
            .expect("gen segmented sort");
    let kernel = load_kernel(&ptx, &format!("segmented_sort_u32_bs{block}"));
    let stream = fx.stream();

    let d_data = DeviceBuffer::<u32>::from_host(&data).expect("d_data");
    let d_off = DeviceBuffer::<u64>::from_host(&offsets).expect("d_off");

    // One block per segment.
    kernel
        .launch(
            &params(num_segments as u32, block),
            &stream,
            &(
                d_data.as_device_ptr(),
                d_off.as_device_ptr(),
                num_segments as u64,
            ),
        )
        .expect("launch segmented sort");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; data.len()];
    d_data.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "segmented_sort per-segment ascending");
}

// ===========================================================================
// radix sort  (LSD: per-pass count -> global-offset scan -> scatter)
// ===========================================================================

use super::GpuFixture;

/// Run a full LSD radix sort by orchestrating the count / scan / scatter
/// kernels over `passes` digit passes with ping-pong key buffers. Returns the
/// fully sorted keys.
#[allow(clippy::too_many_arguments)]
fn radix_full_sort(
    fx: &GpuFixture,
    k_count: &Kernel,
    k_scan: &Kernel,
    k_scatter: &Kernel,
    input: &[u32],
    block: u32,
    radix: u32,
    passes: u32,
    bits: u32,
) -> Vec<u32> {
    let n = input.len();
    let num_blocks = grid_1d(n as u32, block);
    let counts_len = (num_blocks * radix) as usize;
    let stream = fx.stream();

    let d_a = DeviceBuffer::<u32>::from_host(input).expect("d_a");
    let d_b = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("d_b");
    let d_counts = DeviceBuffer::<u32>::from_host(&vec![0u32; counts_len]).expect("d_counts");

    let mut src_is_a = true;
    for pass in 0..passes {
        let shift = pass * bits;
        let (src, dst) = if src_is_a { (&d_a, &d_b) } else { (&d_b, &d_a) };
        k_count
            .launch(
                &params(num_blocks, block),
                &stream,
                &(
                    d_counts.as_device_ptr(),
                    src.as_device_ptr(),
                    n as u64,
                    shift,
                ),
            )
            .expect("launch count");
        // One block of `radix` threads (thread d scans digit d across blocks).
        k_scan
            .launch(
                &params(1, radix),
                &stream,
                &(d_counts.as_device_ptr(), num_blocks),
            )
            .expect("launch scan");
        k_scatter
            .launch(
                &params(num_blocks, block),
                &stream,
                &(
                    dst.as_device_ptr(),
                    src.as_device_ptr(),
                    d_counts.as_device_ptr(),
                    n as u64,
                    shift,
                ),
            )
            .expect("launch scatter");
        src_is_a = !src_is_a;
    }
    stream.synchronize().expect("sync");

    let final_buf = if src_is_a { &d_a } else { &d_b };
    let mut got = vec![0u32; n];
    final_buf.copy_to_host(&mut got).expect("copy");
    got
}

#[test]
fn radix_sort_4bit_u32_sorts() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize;
    let mut rng = Lcg::new(0x_4AD1_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.next_u32() % 1_000_000).collect();
    let mut expected = input.clone();
    expected.sort_unstable();

    let (c, s, sc) = RadixSortTemplate::new(RadixSortConfig {
        ty: PtxType::U32,
        block_size: block,
    })
    .generate(fx.sm)
    .expect("gen radix4");
    let kc = load_kernel(&c, &format!("radix_count_u32_bs{block}"));
    let ks = load_kernel(&s, "radix_scan_u32");
    let ksc = load_kernel(&sc, &format!("radix_scatter_u32_bs{block}"));

    let got = radix_full_sort(&fx, &kc, &ks, &ksc, &input, block, 16, 8, 4);
    assert_eq!(got, expected, "radix 4-bit u32");
}

#[test]
fn radix_sort_8bit_u32_sorts() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize;
    let mut rng = Lcg::new(0x_8AD8_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.next_u32() % 1_000_000).collect();
    let expected = reference_radix8_sort_u32(&input);

    let (c, s, sc) =
        RadixSort8Template::new(RadixSort8Config::new(PtxType::U32, block).expect("cfg"))
            .generate(fx.sm)
            .expect("gen radix8");
    let kc = load_kernel(&c, &format!("radix8_count_u32_bs{block}"));
    let ks = load_kernel(&s, "radix8_scan_u32");
    let ksc = load_kernel(&sc, &format!("radix8_scatter_u32_bs{block}"));

    // 8-bit digits: radix 256, 4 passes for u32.
    let got = radix_full_sort(&fx, &kc, &ks, &ksc, &input, block, 256, 4, 8);
    assert_eq!(got, expected, "radix 8-bit u32");
}

#[test]
fn radix_sort_pairs_ascending_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let n = 3000usize;
    let mut rng = Lcg::new(0x_5A1B_0001);
    let keys: Vec<u32> = (0..n).map(|_| rng.next_u32() % 500_000).collect();
    let vals: Vec<u32> = keys.iter().map(|&k| k ^ 0x1234_5678).collect();
    let k64: Vec<u64> = keys.iter().map(|&k| u64::from(k)).collect();
    let v64: Vec<u64> = vals.iter().map(|&v| u64::from(v)).collect();
    let (exp_k, exp_v) = reference_sort_pairs_by_key(&k64, &v64, SortOrder::Ascending);
    let exp_k: Vec<u32> = exp_k.iter().map(|&k| k as u32).collect();
    let exp_v: Vec<u32> = exp_v.iter().map(|&v| v as u32).collect();

    // count + scatter come from the pairs template; the per-(block,digit) scan
    // is the shared 4-bit radix scan.
    let (count_ptx, scatter_ptx) = RadixPairsTemplate::new(
        RadixPairsConfig::new(PtxType::U32, PtxType::U32, SortOrder::Ascending, block)
            .expect("cfg"),
    )
    .generate(fx.sm)
    .expect("gen pairs");
    let (_, scan_ptx, _) = RadixSortTemplate::new(RadixSortConfig {
        ty: PtxType::U32,
        block_size: block,
    })
    .generate(fx.sm)
    .expect("gen radix4 scan");
    let kc = load_kernel(&count_ptx, &format!("radix_pairs_count_asc_u32_bs{block}"));
    let ks = load_kernel(&scan_ptx, "radix_scan_u32");
    let ksc = load_kernel(
        &scatter_ptx,
        &format!("radix_pairs_scatter_asc_u32_u32_bs{block}"),
    );
    let stream = fx.stream();

    let num_blocks = grid_1d(n as u32, block);
    let counts_len = (num_blocks * 16) as usize;
    let ka = DeviceBuffer::<u32>::from_host(&keys).expect("ka");
    let kb = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("kb");
    let va = DeviceBuffer::<u32>::from_host(&vals).expect("va");
    let vb = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("vb");
    let counts = DeviceBuffer::<u32>::from_host(&vec![0u32; counts_len]).expect("counts");

    let mut src_is_a = true;
    for pass in 0..8u32 {
        let shift = pass * 4;
        let (ks_src, ks_dst, vs_src, vs_dst) = if src_is_a {
            (&ka, &kb, &va, &vb)
        } else {
            (&kb, &ka, &vb, &va)
        };
        kc.launch(
            &params(num_blocks, block),
            &stream,
            &(
                counts.as_device_ptr(),
                ks_src.as_device_ptr(),
                n as u64,
                shift,
            ),
        )
        .expect("count");
        ks.launch(
            &params(1, 16),
            &stream,
            &(counts.as_device_ptr(), num_blocks),
        )
        .expect("scan");
        ksc.launch(
            &params(num_blocks, block),
            &stream,
            &(
                ks_dst.as_device_ptr(),
                vs_dst.as_device_ptr(),
                ks_src.as_device_ptr(),
                vs_src.as_device_ptr(),
                counts.as_device_ptr(),
                n as u64,
                shift,
            ),
        )
        .expect("scatter");
        src_is_a = !src_is_a;
    }
    stream.synchronize().expect("sync");

    let (fk, fv) = if src_is_a { (&ka, &va) } else { (&kb, &vb) };
    let mut got_k = vec![0u32; n];
    let mut got_v = vec![0u32; n];
    fk.copy_to_host(&mut got_k).expect("copy k");
    fv.copy_to_host(&mut got_v).expect("copy v");
    assert_eq!(got_k, exp_k, "radix pairs keys");
    assert_eq!(got_v, exp_v, "radix pairs vals");
}

#[test]
fn onesweep_sort_u32_matches_reference() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let block = 256u32;
    let radix_bits = 8u32;
    let radix = 1usize << radix_bits; // 256
    let mask = (radix as u32) - 1;
    // Kept small: the decoupled-lookback chain spin-waits block-by-block, so a
    // few blocks keep the runtime modest while still exercising cross-block
    // lookback (FLAG_A walk past an intermediate block to a FLAG_P predecessor).
    let n = 900usize;
    let mut rng = Lcg::new(0x_05E3_0001);
    let input: Vec<u32> = (0..n).map(|_| rng.next_u32() % 800_000).collect();
    let expected = reference_onesweep_sort_u32(&input, radix_bits, block);

    let ptx =
        OnesweepTemplate::new(OnesweepConfig::new(PtxType::U32, radix_bits, block).expect("cfg"))
            .generate(fx.sm)
            .expect("gen onesweep");
    let kernel = load_kernel(&ptx, &format!("onesweep_pass_r8_u32_bs{block}"));
    let stream = fx.stream();

    let num_blocks = grid_1d(n as u32, block);
    let scratch = (num_blocks as usize) * radix;
    let d_a = DeviceBuffer::<u32>::from_host(&input).expect("a");
    let d_b = DeviceBuffer::<u32>::from_host(&vec![0u32; n]).expect("b");
    let d_agg = DeviceBuffer::<u32>::from_host(&vec![0u32; scratch]).expect("agg");
    let d_prefix = DeviceBuffer::<u32>::from_host(&vec![0u32; scratch]).expect("prefix");

    let passes = 32 / radix_bits; // 4
    let mut src_is_a = true;
    for pass in 0..passes {
        let shift = pass * radix_bits;
        let (src, dst) = if src_is_a { (&d_a, &d_b) } else { (&d_b, &d_a) };

        // Global digit base = exclusive prefix of the global histogram of the
        // CURRENT keys (read back so we never assume the GPU is correct).
        let mut cur = vec![0u32; n];
        src.copy_to_host(&mut cur).expect("copy cur");
        let mut hist = vec![0u32; radix];
        for &k in &cur {
            hist[((k >> shift) & mask) as usize] += 1;
        }
        let mut gbase = vec![0u32; radix];
        let mut run = 0u32;
        for d in 0..radix {
            gbase[d] = run;
            run += hist[d];
        }

        let d_gbase = DeviceBuffer::<u32>::from_host(&gbase).expect("gbase");
        // status must start at FLAG_X (=0) every pass.
        let d_status =
            DeviceBuffer::<u32>::from_host(&vec![0u32; num_blocks as usize]).expect("st");

        kernel
            .launch(
                &params(num_blocks, block),
                &stream,
                &(
                    dst.as_device_ptr(),
                    src.as_device_ptr(),
                    d_gbase.as_device_ptr(),
                    d_status.as_device_ptr(),
                    d_agg.as_device_ptr(),
                    d_prefix.as_device_ptr(),
                    n as u64,
                    num_blocks,
                    shift,
                ),
            )
            .expect("launch onesweep");
        stream.synchronize().expect("sync");
        src_is_a = !src_is_a;
    }

    let final_buf = if src_is_a { &d_a } else { &d_b };
    let mut got = vec![0u32; n];
    final_buf.copy_to_host(&mut got).expect("copy");
    assert_eq!(got, expected, "onesweep sort");
}
