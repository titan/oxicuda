# oxicuda-levelzero TODO

Intel Level Zero / oneAPI compute backend with OpenCL SPIR-V kernel generators
and dispatch framework. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 6,216 across 10 files
- **Tests:** 103 passing
- **Status:** Full memory + compute, OpenCL SPIR-V generators, XMX cooperative-matrix, sub-group ops, multi-tile dispatcher
- **Targets:** Intel Xe-LP (Gen12), Xe-HPG (Arc Alchemist/Battlemage), Xe-HPC (Ponte Vecchio / Data Center GPU Max)

### Completed

#### Core Infrastructure
- [x] `lib.rs` -- module wiring, re-exports `LevelZeroBackend`, `LevelZeroError`, `LevelZeroResult`
- [x] `backend.rs` -- `LevelZeroBackend` implementing `ComputeBackend`; dispatch for GEMM, unary/binary/reduction, conv2d, attention, batched GEMM (~61.5K, split into `backend/` subdir)
- [x] `device.rs` -- Level Zero driver/device enumeration via runtime-loaded `libze_loader.so`; physical-device + sub-device queries (~28.0K)
- [x] `error.rs` -- `LevelZeroError` (NotInitialized, LibraryNotFound, DeviceNotFound, ShaderCompilation, AllocationError, KernelLaunchError) with thiserror
- [x] `memory.rs` -- `LevelZeroMemoryManager` over `zeMemAllocDevice`/`zeMemFree`/`zeCommandListAppendMemoryCopy`; per-context memory pools (~23.7K)

#### OpenCL SPIR-V Generators (`spirv.rs`, ~48.5K)
- [x] `SpvModule` builder -- emits SPIR-V 1.2 binaries using `Kernel` execution model + `Physical64`/`OpenCL` memory model
- [x] `gemm_compute_shader` -- tiled OpenCL kernel `__kernel void gemm_f32` with `CrossWorkgroup` pointers
- [x] `batched_gemm_compute_shader` -- batch index from `WorkgroupId.z`, stride-based per-batch offsets
- [x] `unary_compute_shader` -- relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu (each via `OpExtInst` OpenCL extended instructions)
- [x] `binary_compute_shader` -- add, sub, mul, div, max, min, pow
- [x] `reduce_compute_shader` -- sum, max, min, mean

#### Neural-Network Kernels (`spirv_nn.rs`, ~25.9K)
- [x] `conv2d_spirv` -- NCHW conv2d kernel (each work-item produces one output element)
- [x] Attention SPIR-V -- scaled dot-product + stable softmax + causal masking
- [x] All dim constants baked as `OpConstant` for inliner optimization

#### Intel XMX Cooperative Matrix (`spirv_xmx.rs`, ~42.5K)
- [x] `XmxTileConfig` -- tile dimensions for Xe-HPG (Arc) and Xe-HPC (Ponte Vecchio) XMX engines
- [x] `gemm_xmx_spirv` -- `SPV_KHR_cooperative_matrix` GEMM (`C = alpha*A*B + beta*C`)
- [x] `gemm_xmx_f16_spirv` -- FP16 input / FP32 accumulation (target Arc XMX)
- [x] `matmul_xmx_bf16_spirv` -- BF16 input / FP32 accumulation (target Xe-HPC)
- [x] Emits `OpCooperativeMatrixLoadKHR` / `MulAddKHR` / `StoreKHR` (opcodes 4456-4459)
- [x] `CooperativeMatrixKHR` capability (6022) + `Shader` (1) + `Float16` (9) capabilities

#### Sub-group Operations (`spirv_subgroup.rs`, ~35.4K)
- [x] `reduction_subgroup_spirv` -- two-phase sub-group reduction via `OpGroupNonUniformFAdd` with `Reduce` group operation
- [x] `scan_subgroup_spirv` -- inclusive prefix sum via `OpGroupNonUniformIAdd` with `InclusiveScan`
- [x] `gemm_subgroup_spirv` -- GEMM with sub-group shuffle (`OpGroupNonUniformShuffle`) for A-row broadcast
- [x] Capability flags: `Addresses` (4), `Kernel` (6), `GroupNonUniform` (61), `GroupNonUniformArithmetic` (63), `GroupNonUniformShuffle` (65)

#### Multi-Tile / Multi-Device (`multi_tile.rs`, ~11.3K)
- [x] `SubDeviceInfo` -- per-tile index, name, EU/XVE execution-unit count
- [x] `WorkDistribution` enum -- `EvenSplit` (default) and `RowSlab { rows_per_tile }`
- [x] `TileWorkSlice` -- row range assigned to each tile
- [x] `MultiTileDispatcher` -- discovers Xe-HPC sub-devices via `zeDeviceGetSubDevices`; transparently falls back to single-device on Arc / Xe-LP

### Future Enhancements

#### P0 -- Critical
- [ ] ESIMD (Explicit SIMD) intrinsics -- emit `OpenCL.std::sub_group_block_read/write` for high-throughput tile loads (critical for Xe-HPC GEMM)
- [ ] Multi-tile slab decomposition for >1 tile on Ponte Vecchio (currently classified but not dispatched cross-tile)
- [ ] FP8 cooperative-matrix on Xe-HPC (`SPV_INTEL_subgroups`) -- BF8/HF8 inference path

#### P1 -- Important
- [ ] IPC handles (`zeMemGetIpcHandle` / `zeMemOpenIpcHandle`) for cross-process device-buffer sharing
- [ ] Bindless / large-array descriptor (`zeKernelSetIndirectAccess`) for very-wide tensor lists
- [ ] Module caching (`zeModuleGetNativeBinary`) -- persist compiled L0 modules across runs
- [ ] Async command-queue groups -- discover and use copy queues separately from compute queues
- [ ] Event-based dependency graph (`ze_event_handle_t`) for inter-kernel synchronization
- [ ] `zeFenceQueryStatus` polling for non-blocking completion detection

#### P2 -- Nice-to-Have
- [ ] `command/command_list_reuse.rs` — command-list reset + reuse API (L0 spec §3.5.6): reset a completed `ze_command_list_handle_t` with `zeCommandListReset` and re-record kernel dispatches for repeated launches without re-allocation; `ReusableCommandList`
- [ ] `device/eu_occupancy.rs` — EU occupancy hints via sysman (`zes_device_handle_t`): query active EU utilisation fraction via `zesDeviceProcessesGetState` and feed back into tile-size selection heuristic; `EuOccupancyAdvisor`
- [ ] `spirv/dpas_gemm.rs` — Intel Arc DPAS (Dot-Product Accumulate Systolic) GEMM (Intel 2022): emit `OpIMad` / `OpDPAS` intrinsics via `SPV_INTEL_subgroups` extension for systolic-array-accelerated INT8/BF16 GEMM on Xe-HPG; `DpasGemm`
- [ ] oneCCL collectives integration (`libccl.so`) -- AllReduce/AllGather across multi-GPU
- [ ] oneMKL GEMM interop (runtime-loaded `libonemkl.so`) for tuned Xe-HPC paths
- [ ] oneDNN primitive cache integration for fused conv+bias+relu
- [ ] DPC++ runtime interop documentation (out-of-process invocation)
- [ ] Level Zero performance metrics API (`zet_metric_*`) for profiling
- [ ] Sparse matrix XMX path (`SPV_INTEL_2:4_sparse`) for structured sparsity

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| libloading | Runtime loading of `libze_loader.so` / `ze_loader.dll` | Yes (implicit via backend) |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 103 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

Intel GPU performance varies by tile architecture and EU count. Xe-HPC targets data-center workloads; Xe-HPG/Arc targets consumer + workstation.

| GPU | Architecture | EUs | XMX | Peak FP16 (TFLOPS) | Peak BF16 | Memory |
|-----|--------------|-----|-----|--------------------|-----------|--------|
| Iris Xe (i7-1185G7) | Xe-LP (Gen12) | 96 | No | ~2.0 | -- | Shared system RAM |
| Arc A380 | Xe-HPG (DG2-128) | 128 | Yes | ~4.5 | ~4.5 | 6 GB GDDR6 |
| Arc A750 | Xe-HPG (DG2-512) | 448 | Yes | ~17.2 | ~17.2 | 8 GB GDDR6 |
| Arc A770 | Xe-HPG (DG2-512) | 512 | Yes | ~19.7 | ~19.7 | 16 GB GDDR6 |
| Arc B580 (Battlemage) | Xe2-HPG | 160 | Yes | ~24.0 | ~24.0 | 12 GB GDDR6 |
| Data Center GPU Max 1100 | Xe-HPC (Ponte Vecchio) | 56 Xe-cores | Yes | ~52 | ~52 | 48 GB HBM2e |
| Data Center GPU Max 1550 | Xe-HPC (2 tiles) | 128 Xe-cores | Yes | ~104 | ~104 | 128 GB HBM2e |

- **GEMM (current scalar SPIR-V)**: target ≥ 60% of oneMKL on Arc A770
- **GEMM (with XMX cooperative-matrix)**: target ≥ 85% of oneMKL on Arc A770
- **XMX BF16 GEMM**: target ≥ 80% of oneMKL on PVC (Xe-HPC)
- **Sub-group reduction**: target ≥ 95% of theoretical bandwidth via single-pass SIMD reduction
- **Kernel dispatch overhead**: target < 15 µs above raw `zeCommandListAppendLaunchKernel`

## Notes

- All Level Zero calls go through runtime-loaded `libze_loader.so` / `ze_loader.dll` -- no link-time dependency
- macOS builds compile but return `LevelZeroError::UnsupportedPlatform` (Apple doesn't ship Intel discrete drivers)
- Linux (Intel GPU) requires `intel-compute-runtime` (oneAPI) installed; tested with v23+
- Windows (Intel GPU) requires Intel oneAPI Base Toolkit drivers (DCH/oneAPI)
- SPIR-V generator targets version 1.2 for maximum Level Zero compatibility (1.6 needed only for cooperative matrix)
- XMX kernels require `CooperativeMatrixKHR` capability + driver support (Arc / Xe-HPC)
- OpenCL execution model (`Kernel` not `GLCompute`) -- distinct from Vulkan SPIR-V despite shared opcode set

## Architecture-Specific Deepening Opportunities

### Xe-LP (Gen12 integrated, Tiger Lake/Alder Lake/Raptor Lake iGPU)
- [ ] EU subgroup size 8/16/32 negotiation per kernel
- [ ] Shared-system-memory tile sizes (no dedicated VRAM)
- [ ] Documented as functional-only (low FP16 throughput, no XMX)
- [ ] LP cache-aware tile sizing for L3-resident workloads

### Xe-HPG (Arc Alchemist DG2: A310/A380/A580/A750/A770)
- [ ] XMX matrix-engine integration via `SPV_KHR_cooperative_matrix` (FP16/BF16)
- [ ] Battlemage (Xe2-HPG) updated XMX shapes when driver support lands
- [ ] DG2 16x16x16 FP16 tile primitives
- [ ] Subgroup-size 32 explicit declaration for Arc compute

### Xe-HPC (Ponte Vecchio / Data Center GPU Max 1100 / 1550)
- [ ] Multi-stack (multi-tile) compute on Max 1550 (2 tiles, 128 Xe-cores total)
- [ ] HBM2e bandwidth-aware tile sizing
- [ ] BF16 XMX path validated against oneMKL
- [ ] Larger tile shapes (32x32) supported on PVC vs Arc
- [ ] EU mode 1 (FP64 fast path) for scientific workloads

### Xe2-HPC / Falcon Shores (future)
- [ ] Hypothetical FP8 XMX shapes (FP8 e4m3 / e5m2) -- placeholder for hardware availability

### Cross-Generation
- [ ] EU subgroup width detection: 8 (older), 16 (Xe-LP), 32 (Xe-HPG/HPC default)
- [ ] Pre-XMX vs XMX dispatch decision based on `zeDeviceGetProperties` capabilities

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These items represent the gap between current implementation depth and production-grade Level Zero parity.

### Test Coverage Gaps
- [ ] Multi-tile correctness on PVC 1550 (2-tile system) -- currently single-tile dispatch only
- [ ] XMX numerical accuracy vs oneMKL reference (FP16, BF16) -- currently SPIR-V generation tests only
- [ ] ESIMD intrinsic emission tested when implemented (no current test coverage)
- [ ] Sub-group shuffle GEMM correctness vs scalar GEMM reference
- [ ] IPC handle round-trip across two processes (when implemented)
- [ ] Module-binary cache reuse across runs (when implemented)
- [ ] CI matrix across Xe-LP (iGPU), Xe-HPG (Arc), Xe-HPC (PVC) -- currently only emulator/spec tests

### Implementation Deepening
- [ ] In-house SPIR-V builder fuzz-tested vs `spirv-val` (validates OpenCL execution model emissions)
- [ ] OpenCL extended-instruction-set coverage extended (currently relu/gelu/exp/log; add atan2, erf, etc.)
- [ ] Async copy queues discovered and used (currently single command-queue family)
- [ ] `zeFenceCreate` instead of `zeCommandListReset` for finer-grained completion tracking
- [ ] Multi-context support (currently single root context) for multi-process workloads
- [ ] Driver upgrade negotiation -- pre-1.5 vs 1.5+ feature gating (XMX requires 1.5+)
- [ ] EU power-state hinting via `zesPowerSetLimits` for thermal-aware dispatch
- [ ] L0 sysman (`zes_*`) integration for telemetry (utilization, temperature, ECC)

## Level Zero Version Compatibility

| L0 Spec | intel-compute-runtime | Status | Notes |
|---------|----------------------|--------|-------|
| 1.3 | 22.x | Build only | Pre-XMX support era |
| 1.4 | 23.x | Tested | XMX path negotiates feature flag |
| 1.5 | 23.40+ | Tested | `SPV_KHR_cooperative_matrix` widely deployed |
| 1.6 | 24.x | Verified | Default target -- sub-device queries stable |
| 1.7 | 24.40+ | Tested | PVC multi-tile sub-device default |

- Library candidates: `libze_loader.so.1`, `libze_loader.so`, `ze_loader.dll` -- searched in order
- **OpenCL SPIR-V version**: 1.2 default (max compatibility); 1.6 conditional for `cooperative_matrix_KHR`
- **DPC++ / SYCL** not a dependency; this crate stays at the L0 layer below SYCL

## Observability & Diagnostics

- [ ] `tracing` spans on every `LevelZeroBackend::*` entry point with kernel + global-size
- [ ] Optional `--features sysman` enables `zes_*` telemetry (utilization, temperature, ECC, power)
- [ ] L0 module-compile log captured in `LevelZeroError::ShaderCompilation` (currently swallows compiler output)
- [ ] Per-tile dispatch utilization metric for multi-tile PVC scheduling decisions
- [ ] Validation-layer integration via `ZE_ENABLE_VALIDATION_LAYER=1` env var when present

## Roadmap & Milestones

- **v0.2 (Xe-HPC readiness)**: ESIMD intrinsics, multi-tile slab decomp, FP8 XMX path
- **v0.3 (Ecosystem)**: IPC handles, module-binary cache, oneCCL collectives integration
- **v0.4 (Polish)**: oneMKL GEMM interop, performance metrics API, sysman telemetry
- **v1.0 (Stable)**: oneMKL / oneDNN parity on Xe-HPC (PVC), full multi-tile Arc B-series Battlemage support
