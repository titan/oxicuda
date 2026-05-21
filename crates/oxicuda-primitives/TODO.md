# oxicuda-primitives TODO

CUB-equivalent parallel GPU primitives with zero CUDA SDK dependency. All kernels are generated as PTX source strings at runtime and JIT-compiled via `cuModuleLoadData`. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/C++, no nvcc, no CUB headers.

## Implementation Status

**Actual: 6,216 SLoC across 18 files** (4 sub-crates: `warp/`, `block/`, `device/`, `sort/` + shared `ptx_helpers.rs`, `handle.rs`, `error.rs`)

Coverage spans the full CUB hierarchy: warp-level reductions / scans built on `shfl.sync.*`, block-level reductions / scans built on shared memory + warp shuffles, device-wide reduce / scan / stream-compaction / histogram pipelines, and two sort algorithms (4-bit LSD radix sort, bitonic-block + co-rank merge sort). Every public template generates target-specific PTX for sm_75 through sm_120 via `oxicuda-ptx`.

### Completed [x]

#### Core infrastructure
- [x] `error.rs` (207 SLoC) -- `PrimitivesError` with 9 variants (`Cuda(#[from] CudaError)`, `BufferTooSmall`, `InputTooLarge`, `InvalidArgument`, `UnsupportedOperation`, `PtxGeneration`, `KernelLoad`, `KernelLaunch`, `DimensionMismatch`, `WorkspaceTooSmall`); convenience constructors `ptx()`, `load()`, `launch()`; `PrimitivesResult<T>` alias; 8 unit tests
- [x] `handle.rs` (160 SLoC) -- `PrimitivesHandle { ctx: Arc<Context>, stream: Arc<Stream>, sm: SmVersion }`; `new()`, `from_arc()`, `sm_version()`, `context()`, `stream()`; `cc_to_sm()` mapping (7.5/8.0/8.6/8.9/9.0/10.0/12.0 → Sm75/Sm80/Sm86/Sm89/Sm90/Sm100/Sm120, fallback Sm80); 3 unit tests
- [x] `lib.rs` (46 SLoC) -- module wiring and top-level re-exports (`PrimitivesError`, `PrimitivesResult`, `PrimitivesHandle`, `PrimitiveType`, `ReduceOp`) plus a doc-test that round-trips a `DeviceReduceTemplate` PTX generation

#### PTX code-generation helpers (`ptx_helpers.rs`, 499 SLoC)
- [x] `ptx_header(sm: SmVersion)` -- `.version` / `.target` lines including the Sm90a, Sm100, Sm120 cases
- [x] `ptx_type_str(PtxType)` / `ptx_type_bytes(PtxType)` -- full coverage of every variant defined in `oxicuda-ptx::ir::PtxType`: B8/B16/B32/B64/B128, U8/U16/U32/U64, S8/S16/S32/S64, F16/F16x2, BF16/BF16x2, F32, TF32, F64, E4M3/E5M2 (FP8), E2M3/E3M2 (FP6 variants), E2M1 (FP4), Pred
- [x] `reg_decl(reg_prefix, ty, count)` -- helper for emitting `.reg .ty %reg<N>;` blocks
- [x] `ReduceOp` enum: Sum, Product, Min, Max, And, Or, Xor (7 ops); `mnemonic_for(ty)`, `identity_literal(ty)` per op × type
- [x] `PrimitiveType` trait -- bridges Rust scalar types to `PtxType` and byte-width for templated kernel generators
- [x] 16 unit tests covering every SmVersion target string, every PtxType byte width, and every ReduceOp identity / mnemonic

#### Warp-level primitives (`warp/`, 861 SLoC)
- [x] `warp/reduce.rs` (439 SLoC) -- `WarpReduceConfig`, `WarpReduceTemplate`; `shfl.sync.bfly.b32` butterfly tree (5 steps for 32-lane warp); f64 split into lo/hi 32-bit registers and recombined; optional broadcast lane mask to write the result back to a chosen lane; all 7 reduce ops × all numeric types
- [x] `warp/scan.rs` (405 SLoC) -- `WarpScanConfig`, `WarpScanTemplate`, `ScanKind { Inclusive, Exclusive }`; `shfl.sync.up.b32` shift-and-combine (5 steps); f64 lo/hi split; exclusive variant subtracts the seed
- [x] 22 unit tests across reduce + scan (PTX string content checks, every op × type × SmVersion)

#### Block-level primitives (`block/`, 940 SLoC)
- [x] `block/reduce.rs` (476 SLoC) -- `BlockReduceConfig`, `BlockReduceTemplate`, `MAX_BLOCK_SIZE`; two-stage algorithm: per-warp reduction via `shfl.sync.bfly`, partial sums spilled to shared memory, warp 0 reduces the partials, broadcast back to all threads
- [x] `block/scan.rs` (464 SLoC) -- `BlockScanConfig`, `BlockScanTemplate`; work-efficient Blelloch algorithm with explicit up-sweep / down-sweep over shared memory; inclusive + exclusive; f64 split throughout
- [x] 26 unit tests across reduce + scan (block sizes 32/64/128/256/512/1024, every op × type)

#### Device-wide primitives (`device/`, 2,256 SLoC)
- [x] `device/reduce.rs` (579 SLoC) -- `DeviceReduceConfig`, `DeviceReduceTemplate`, `DEFAULT_BLOCK_SIZE`; 2-pass pipeline -- pass 1 emits per-block partials, pass 2 reduces the partials to a single scalar; all 7 ReduceOps × all numeric types × all SM versions; 11 unit tests
- [x] `device/scan.rs` (645 SLoC) -- `DeviceScanConfig`, `DeviceScanTemplate`; 3-kernel pipeline -- (1) per-block exclusive scan, (2) propagate block aggregates, (3) apply block prefix to local results; inclusive + exclusive; 8 unit tests
- [x] `device/select.rs` (487 SLoC) -- `DeviceSelectConfig`, `DeviceSelectTemplate`, `SelectPredicate { NonZero, Positive, Negative, FlagArray }`; 2-kernel flag+gather pipeline implemented on top of exclusive scan; type-correct `setp.{lt,gt,ne}.{ty}` per predicate × element type; unsigned types short-circuit `Negative` to always-false; 10 unit tests
- [x] `device/histogram.rs` (645 SLoC) -- `DeviceHistogramConfig`, `DeviceHistogramMode { Modulo, EvenRange }`, `DeviceHistogramTemplate`; 2-kernel init+count pipeline; per-block shared-memory privatized histogram with `atom.shared.add.u32` for collision-free per-bin updates; strided global merge; integer (`rem.u32` / `rem.u64`) and fp/integer linear `EvenRange` mappings; 11 unit tests

#### Sort primitives (`sort/`, 1,102 SLoC)
- [x] `sort/radix_sort.rs` (524 SLoC) -- `RadixSortConfig`, `RadixSortTemplate`; 4-bit LSD radix sort; 8 passes for u32 keys, 16 for u64; three kernels per pass:
  - **Count** -- private `cnt_hist[16]` in shared memory + `atom.shared.add.u32`
  - **Scan** -- single block × 16 threads doing a sequential column scan for the exclusive prefix
  - **Scatter** -- `block_offs[16]` pre-loaded, `atom.shared.add.u32` for unique output positions, ping-pong between two key buffers between passes
  - 12 unit tests
- [x] `sort/merge_sort.rs` (578 SLoC) -- `MergeSortConfig`, `MergeSortTemplate`; stable bitonic block sort + co-rank merge sort:
  - **Bitonic sort** -- 2-barrier-per-stage correctness pattern (pre-load + pre-write); `selp.{ty}` for type-correct compare-swap
  - **Merge kernel** -- O(log n) co-rank binary search per output element; branch-based output selection
  - All 11 unit tests passing including 142-thread blocks and odd-length inputs

### Future Enhancements [ ]

#### P0 -- Pipeline depth and bandwidth (CUB parity)
- [x] Privatized histogram bin contention -- `atom.shared.add.u32` already in place
- [x] Radix sort scatter unique-position guarantee via `block_offs[16]` + atomic
- [ ] Decoupled-lookback scan -- replace the 3-kernel `device/scan.rs` pipeline with the single-kernel decoupled-lookback algorithm used by modern CUB (significantly fewer kernel launches, much higher bandwidth)
- [ ] 8-bit radix per pass with shared-memory shared exclusive prefix (CUB default) -- would halve the number of passes vs the current 4-bit implementation but increases shared-memory pressure
- [ ] Onesweep radix sort -- single-kernel global decoupled lookback, removing the per-pass scan kernel entirely

#### P1 -- Algorithmic coverage
- [ ] `DeviceRunLengthEncode` -- compact identical-value runs (used by sparse formats and tokenizers)
- [ ] `DeviceSegmentedReduce` / `DeviceSegmentedScan` -- per-segment aggregates given segment offsets (needed by GraphRS, dynamic batching)
- [ ] `DeviceSegmentedSort` / `DeviceSegmentedRadixSort` -- per-segment sort
- [ ] `DeviceSelectUnique` -- compact consecutive duplicates (paired with the existing `DeviceSelect` flag-mode predicate)
- [ ] `DevicePartition` -- two-output flag-based partition (positive predicate matches → buf A, negative → buf B)
- [ ] `DeviceMergeKeysValues` -- key+value parallel merge for stable join operations

#### P1 -- Sort completeness
- [ ] Key+value variants for both radix and merge sort (currently both are keys-only)
- [ ] Descending order option (currently radix sort emits ascending only)
- [ ] f32/f64 radix sort via sign-flipping pre-pass to reinterpret as unsigned (already supported in CUB)
- [ ] Pair-key sort (struct-of-arrays) for joint sorting of two keys with combined ordering

#### P2 -- Architecture-aware templates
- [ ] sm_90 cluster-launch reductions -- use `cp.async.bulk` to stage block partials directly into a peer block's shared memory
- [ ] sm_100 / sm_120 thread-block clusters for device-wide scan -- single-cluster scan with `barrier.cluster.sync`
- [ ] Tensor Memory Accelerator (TMA) descriptors for histogram bulk-loads on Hopper+

#### P2 -- Quality of life
- [ ] `PrimitivesHandle::auto()` -- single-call helper that drives `oxicuda_driver::init()`, picks the best device, creates a context + stream, and returns a handle (today callers wire 4-5 driver calls themselves)
- [ ] A "host reference" mode that runs the same algorithm on the CPU (over an `ndarray::Array`) for cross-checking GPU output during development
- [ ] Public `workspace_bytes(input_len)` query on every device template so callers can pre-allocate exactly the right scratch buffer

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API loader (`Context`, `Stream`, module loading) | Yes (runtime FFI only) |
| oxicuda-memory | Device buffer allocation for partials / scratch / output | Yes |
| oxicuda-launch | Type-safe kernel launch with `Dim3` grid/block | Yes |
| oxicuda-ptx | PTX IR + code emitter (`SmVersion`, `PtxType`, instruction builders) | Yes |
| thiserror | `#[derive(Error)]` for `PrimitivesError` | Yes |
| tracing | Structured logging on PTX generation + launch | Yes |
| approx (dev) | Floating-point comparison in tests | Yes |

## Quality Status

- Warnings: 0
- Tests: 142 unit tests + 12 doctests (154 total, all passing as of the 2026-05-15 root-TODO snapshot)
- unwrap() calls: 0 (production code)
- clippy: clean (pedantic + nursery)
- ptx_helpers coverage: all `PtxType` variants (including B128, E4M3/E5M2/E2M3/E3M2/E2M1, F16x2, BF16x2, TF32), all `SmVersion` variants (Sm90a, Sm100, Sm120), all `ReduceOp` identities and instruction mnemonics

## Performance Targets

| Algorithm | Sizes | sm_80 (A100) Target |
|-----------|-------|---------------------|
| Device reduce, u32 | 1M, 16M, 256M | ≥ 90% of CUB `DeviceReduce::Sum` bandwidth |
| Device scan (inclusive), u32 | 1M, 16M | ≥ 80% CUB `DeviceScan::InclusiveSum` (will rise to ≥ 90% once decoupled-lookback lands) |
| Stream compaction, u32, density 50% | 16M | ≥ 80% CUB `DeviceSelect::If` |
| Histogram, 256 bins, u8 input | 256M | ≥ 90% CUB `DeviceHistogram::HistogramEven` (privatized shared) |
| Radix sort, u32 keys | 8M, 64M | ≥ 75% CUB `DeviceRadixSort::SortKeys` (will rise once onesweep lands) |
| Radix sort, u64 keys | 8M | ≥ 70% CUB |
| Merge sort, u32 keys | 1M | ≥ 60% CUB `DeviceMergeSort::SortKeys` (bitonic+co-rank is asymptotically slower than CUB's tile-based merge but is stable and simpler) |

## Notes

- All kernels are emitted as PTX strings, never as C++ source; nothing in this crate touches `nvcc`, `libnvrtc`, or CUB headers
- The runtime path is `oxicuda-ptx` → PTX string → `oxicuda-driver::Module::from_ptx` → `cuModuleLoadData` → `Function::from_module` → `oxicuda-launch::Kernel::launch`
- macOS builds compile but every GPU-touching call returns `PrimitivesError::Cuda(CudaError::DriverNotAvailable)` at runtime
- f64 is supported throughout via lo/hi 32-bit register splits because `shfl.sync.*` only operates on 32-bit lanes
- The crate intentionally does not depend on `oxicuda-blas` -- primitives are the building blocks for it, not the other way around

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89)
- [ ] `cp.async.cg` (cache global) for staging input tiles into shared memory in the histogram and scan kernels -- currently uses plain `ld.global` so we miss the cache hint
- [ ] 4-stage software pipelining of load / count / scatter in the radix sort scatter kernel to overlap memory and compute
- [ ] Bank-conflict-free shared layout for the histogram private bins (the current 16-bin layout is already conflict-free but a 256-bin variant for `HistogramEven` would need explicit padding)

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [ ] `wgmma`-friendly tile sizes are out of scope here (this crate is non-Tensor-Core) but the sort kernels could use `cp.async.bulk` to bulk-load 1 KiB tiles via TMA
- [ ] Distributed shared memory across a thread-block cluster -- would let device-wide reduce skip the 2-pass pipeline entirely for inputs that fit in one cluster
- [ ] Sm90a `setmaxnreg.async` to widen the register budget in the scatter kernel for u64 sorts

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the gap between the present generator-based implementation and the depth of NVIDIA's CUB.

### Test Coverage Gaps
- [ ] Property-based test: for every `(ReduceOp, PtxType)` pair, the generated PTX contains exactly the expected mnemonic and identity literal (the current tests check a representative sample)
- [ ] Round-trip correctness via the CPU reference: `PrimitivesHandle::reference_reduce(op, input)` compared against the GPU output for `n ∈ {1, 31, 32, 33, 1023, 1024, 1025, 1M}` covering warp / block / cross-block boundary cases
- [ ] Stress test: 10⁶ randomized radix-sort runs with random length, distribution (uniform / Zipfian / sorted / reverse-sorted / all-equal) verifying stability of merge sort and correctness of radix sort
- [ ] Histogram density sweep: 1, 16, 256, 4096 bins × {uniform, skewed} input to catch the shared-memory atomic contention regime

### Implementation Deepening
- [ ] Decoupled-lookback scan rewrite (the single biggest performance gap vs CUB)
- [ ] Onesweep radix sort rewrite (the second-biggest gap)
- [ ] Fused select+scan single-kernel `DeviceSelect` (today the flag pass and the gather pass are separate launches with a scan in between)
- [ ] Templated tile size autotune via `oxicuda-autotune` -- per (algorithm × SM × element type × size class) pick block size and items-per-thread instead of the current `DEFAULT_BLOCK_SIZE = 256` hardwire
- [ ] Documentation: a numbered "CUB → oxicuda-primitives mapping" table mirroring the CUTLASS table in `oxicuda-blas/TODO.md` so consumers can find the right template by CUB name
