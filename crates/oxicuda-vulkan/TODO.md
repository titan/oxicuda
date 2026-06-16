# oxicuda-vulkan TODO

Vulkan compute backend via ash, providing GPU-accelerated operations through
SPIR-V compute shader dispatch. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~5,116 across 22 files
- **Tests:** 86 passing
- **Status:** Full memory + compute backend via in-house SPIR-V builder, multi-queue async, pipeline cache
- **Targets:** Vendor-agnostic (NVIDIA / AMD / Intel / Mesa lavapipe), Vulkan 1.2+

### Completed

#### Core Infrastructure
- [x] `lib.rs` -- module wiring, re-exports `VulkanBackend`, `AsyncComputeManager`, `VulkanFence`, `VulkanSemaphore`
- [x] `backend.rs` -- `VulkanBackend` implementing `ComputeBackend`; dispatch for GEMM, unary/binary/reduction, conv2d, attention, batched GEMM (~59.2K, includes shader-cache and pipeline-cache wiring)
- [x] `device.rs` -- Vulkan instance/device/queue creation via `ash::Entry::load()`; physical device selection with compute-queue family discovery
- [x] `error.rs` -- `VulkanError` (LoadError, NoDevice, NoComputeQueue, ShaderError, AllocationError, CommandBufferError) with thiserror
- [x] `memory.rs` -- `VulkanMemoryManager` over `vkAllocateMemory`/`vkBindBufferMemory`; host-visible + device-local memory types
- [x] `command.rs` -- `VulkanCommandPool` with per-queue-family pools and reset/free semantics
- [x] `pipeline.rs` -- `VulkanComputePipeline` owns `ShaderModule`/`DescriptorSetLayout`/`PipelineLayout`/`Pipeline`/`DescriptorPool`; reverse-order Drop

#### In-house SPIR-V Builder (`spirv/builder.rs`, `spirv/preamble.rs`, `spirv/consts.rs`)
- [x] `SpvModule` builder -- emits valid SPIR-V 1.3 binaries from a typed IR (no rspirv dependency)
- [x] Capabilities, memory model, entry-point, decorations emitted per Vulkan 1.1+ requirements
- [x] All generated shaders use `StorageBuffer` SSBO bindings at descriptor set 0
- [x] Parameters passed via additional SSBO (avoids push-constant size limits)

#### SPIR-V Generators (`spirv/*.rs`)
- [x] `gemm_compute_shader` -- tiled GEMM with workgroup-shared memory, configurable tile size
- [x] `batched_gemm_compute_shader` -- 3D dispatch with batch index from `WorkgroupId.z`
- [x] `unary_compute_shader` -- emits per-op SPIR-V for relu/sigmoid/tanh/exp/log/sqrt/abs/neg/gelu/silu via `UnaryOp` enum
- [x] `binary_compute_shader` -- add/sub/mul/div/max/min/pow via `BinaryOp` enum
- [x] `reduce_compute_shader` -- sum/max/min/mean via `ReduceOp` enum, two-pass workgroup reduction
- [x] `conv2d_spirv` -- NCHW conv2d kernel + CPU fallback for unsupported shapes
- [x] `attention_spirv` -- scaled dot-product attention with stable softmax and causal masking
- [x] `subgroup::reduction_subgroup_spirv` -- warp-level reduction via `GroupNonUniform` opcodes
- [x] `subgroup::scan_subgroup_spirv` -- prefix-sum via `GroupNonUniformIAdd` with `InclusiveScan`
- [x] `trivial::trivial_compute_shader` -- minimal validation shader for plumbing tests

#### Async Compute (`async_compute.rs`, ~13.3K)
- [x] `AsyncComputeManager` -- multi-queue dispatcher with `AtomicUsize` round-robin selection
- [x] `VulkanFence` (host-side wait) and `VulkanSemaphore` (queue-to-queue sync)
- [x] Per-queue command-pool isolation prevents cross-queue mutation hazards
- [x] Timeline semaphore support for ordered async dependency chains
- [x] Multi-queue overlap of compute and transfer queues when hardware provides them

#### Pipeline & Shader Caching (in `backend.rs`)
- [x] `ShaderKey` hash (op type + dtype + shape class) -> `Mutex<HashMap<ShaderKey, Pipeline>>`
- [x] `VkPipelineCache` backing for cross-process pipeline compile reuse
- [x] Eliminates per-dispatch shader recompile overhead (multi-second savings for hot loops)

### Future Enhancements

#### P0 -- Critical
- [ ] `VK_KHR_cooperative_matrix` Tensor-Core-equivalent GEMM -- emit cooperative-matrix SPIR-V (FP16/BF16/FP8) for NVIDIA RTX 30/40, AMD RDNA3, Intel Arc
- [ ] Subgroup-size-control (`VK_EXT_subgroup_size_control`) -- pick 32 (NVIDIA/Intel) vs 64 (AMD GCN/CDNA) vs 32/64 (RDNA) at pipeline creation
- [ ] Push descriptors (`VK_KHR_push_descriptor`) -- bind buffers directly into command buffer without descriptor-set allocation for low-overhead dispatch

#### P1 -- Important
- [ ] Vulkan Memory Allocator equivalent -- sub-allocation from large device-memory blocks (`vk_mem_alloc`-style API in pure Rust)
- [ ] `VK_KHR_dynamic_rendering` integration NOT needed (compute-only), but `VK_KHR_synchronization2` for finer barrier control IS needed
- [ ] Timeline semaphore chains across queues -- multi-queue dependency graphs (currently fence-only per queue)
- [ ] `VK_EXT_shader_atomic_float` for FP32 atomic reductions (replaces two-pass reduction on supporting hardware)
- [ ] Validation layer integration toggle (`VK_LAYER_KHRONOS_validation`) gated behind `validation` feature

#### P2 -- Nice-to-Have
- [ ] `pipeline/vulkan_memory_model.rs` — Vulkan Memory Model explicit acquire-release barriers (Vulkan 1.2 core): emit `OpLoad` / `OpStore` with `MakeAvailable` / `MakeVisible` semantics replacing current global `vkCmdPipelineBarrier`; `VulkanMemModel`
- [ ] `spirv/subgroup_size_control.rs` — `VK_EXT_subgroup_size_control` subgroup-size negotiation (2020): declare fixed `SubgroupSize` at pipeline creation (32 NVIDIA/Intel, 32/64 AMD) for vendor-optimal warp reductions; `SubgroupSizeController`
- [ ] `spirv/performance_query.rs` — `VK_KHR_performance_query` kernel-level GPU timestamps (2020): query pool with `VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR` for per-dispatch GPU cycle counts; `PerformanceQueryPool`
- [ ] `memory/descriptor_buffer.rs` — `VK_EXT_descriptor_buffer` bindless descriptor sets (Vulkan 2023): embed descriptor data directly in device memory for ultra-low-overhead large-model weight binding; `DescriptorBuffer`
- [ ] `VK_EXT_mesh_shader` compute-mesh interop (graphics+compute pipelines) -- not used for ML workloads
- [ ] Ray-query (`VK_KHR_ray_query`) for compute shaders -- enables BVH-based sparse op layouts (research)
- [ ] `VK_KHR_video_*` integration as out-of-scope: explicitly excluded
- [ ] DLSS/FSR scaler shader generation -- left to graphics pipelines
- [ ] `VK_KHR_acceleration_structure` for sparse tensor compaction (research)
- [ ] SPIR-V 1.6 capability path (currently 1.3) -- conditional emit when device supports it

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| ash | Vulkan API bindings + dynamic loader | Yes |
| oxicuda-backend | Common `ComputeBackend` trait | Yes |
| thiserror | Error derive macros | Yes |

## Quality Status

- Warnings: 0
- Tests: 86 passing
- unwrap() calls: 0
- Clippy: clean (pedantic + nursery)

## Performance Targets

Vulkan compute performance varies dramatically by vendor, driver, and feature support. We target competitive performance vs the vendor-native API on each platform.

| Vendor | Reference API | Target Throughput | Notes |
|--------|---------------|-------------------|-------|
| NVIDIA | CUDA / cuDNN | ≥ 80% of CUDA for GEMM 4096³ | Vulkan SPIR-V on RTX 4090 |
| AMD | ROCm / HIP | ≥ 85% of HIP for GEMM 4096³ | Vulkan widely deployed on RDNA |
| Intel Arc | Level Zero | ≥ 90% of L0 for GEMM 4096³ | Same XMX hardware, both paths |
| Mesa lavapipe | CPU reference | Functional only | Software rasterizer |

- **Cooperative-matrix GEMM** (future): target ≥ 90% of vendor-native MMA throughput
- **Subgroup reduction**: target ≥ 95% of theoretical bandwidth via single-pass SIMD reduction
- **Kernel dispatch overhead**: target < 20 µs above raw `vkCmdDispatch` (includes descriptor binding)
- **Pipeline compile**: cached `VkPipelineCache` blob shared across runs for sub-millisecond reuse

## Notes

- All Vulkan calls go through `ash::Entry::load()` -- no link-time dependency on `libvulkan.so`/`vulkan-1.dll`
- macOS Vulkan via MoltenVK -- compiled but `init()` returns `Err` (no native Vulkan on macOS)
- Linux + Vulkan driver 1.2+ required (Mesa 21.0+ for AMD/Intel, NVIDIA proprietary 470+)
- Windows + Vulkan driver 1.2+ required (NVIDIA / AMD / Intel official drivers)
- In-house SPIR-V builder avoids `rspirv` dependency to keep tree minimal and version-controlled
- SPIR-V targets version 1.3 by default (universal Vulkan 1.1+ support)

## Architecture-Specific Deepening Opportunities

### NVIDIA Vulkan (Turing/Ampere/Ada/Hopper)
- [ ] `VK_NV_cooperative_matrix` (vendor extension, pre-KHR) for older drivers
- [ ] `VK_NVX_binary_import` for CUDA-compiled PTX/SASS reuse via Vulkan path
- [ ] Subgroup size 32 explicit declaration
- [ ] FP16/BF16 MMA shapes: 16x16x16 (Turing+), 16x8x16 (Ampere+)

### AMD Vulkan (GCN5/RDNA1/2/3, CDNA via radeonsi)
- [ ] `VK_AMD_shader_explicit_vertex_parameter` (not needed for compute, marker)
- [ ] Wave32 vs Wave64 dispatch via `VK_EXT_subgroup_size_control`
- [ ] RDNA3 WMMA via `VK_KHR_cooperative_matrix`
- [ ] Bank-conflict-free LDS tile layouts (32 banks of 4 bytes each)

### Intel Vulkan (Xe-LP/Xe-HPG/Xe-HPC via Anv)
- [ ] XMX cooperative-matrix path on Arc A-series (mirrors `oxicuda-levelzero/spirv_xmx.rs`)
- [ ] Subgroup size 8/16/32 negotiation via subgroup-size-control
- [ ] `VK_INTEL_shader_integer_functions2` for sub-byte integer ops

### Mesa Lavapipe (CPU software rasterizer)
- [ ] Functional-only testing path for CI without GPU hardware
- [ ] Document expected slowdown (~100x vs hardware) and gate perf tests

### Adreno (Qualcomm mobile, Vulkan 1.3 conformant)
- [ ] Tile sizes tuned for tile-based deferred rendering hardware
- [ ] Subgroup size 32 with FP16 native support (Adreno 6xx+)

### Mali (ARM mobile, Vulkan 1.3 conformant)
- [ ] Subgroup size 16 or 32 depending on Mali generation
- [ ] No bank conflicts (flat shared memory) -- different tile heuristics

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage. These items represent the gap between current implementation depth and production-grade Vulkan compute parity.

### Test Coverage Gaps
- [ ] Vulkan validation-layer-clean CI run on Linux+NVIDIA, Linux+AMD, Linux+Intel separately
- [ ] Multi-queue async correctness on hardware with 2+ compute queues (NVIDIA: 1 compute family, AMD: 1-2, Intel: 1)
- [ ] Cooperative-matrix kernels verified vs scalar reference (when `VK_KHR_cooperative_matrix` is implemented)
- [ ] SPIR-V binary verified by `spirv-val` external tool in CI pre-commit
- [ ] Pipeline-cache reuse across process restarts validated (binary blob portable)
- [ ] Subgroup-size-control negotiated correctly per vendor (NVIDIA 32, AMD 32/64, Intel 8/16/32)

### Implementation Deepening
- [ ] In-house SPIR-V builder fuzz-tested against `spirv-val` for edge cases
- [ ] `VK_EXT_descriptor_indexing` bindless descriptor sets for large model weight tables
- [ ] `VK_KHR_buffer_device_address` for pointer-based buffer access (replaces descriptor indirection)
- [ ] Async pipeline compile via `VK_PIPELINE_CREATE_LIBRARY_BIT_KHR` (compile-while-running)
- [ ] Memory-budget queries (`VK_EXT_memory_budget`) to drive eviction policy
- [ ] Performance counters via `VK_KHR_performance_query` for kernel-level GPU timing

## Vulkan Version Compatibility

| Vulkan API | Minimum Driver | Status | Required Features |
|------------|----------------|--------|-------------------|
| 1.1 | All conformant drivers | Build only | Subgroup ops baseline |
| 1.2 | NVIDIA 450+, AMD 21+, Intel 21+ | Tested | Timeline semaphores, buffer device address, descriptor indexing |
| 1.3 | NVIDIA 510+, AMD 22+, Intel 22+ | Verified | `synchronization2`, dynamic rendering (unused in compute) |
| 1.4 | NVIDIA 555+, AMD 24+ | Future | Push descriptors core, host image copy |

- **ash crate**: `0.38+` required (re-exports SDK 1.3.x function tables)
- **SPIR-V**: emit version 1.3 by default; 1.6 conditionally for `cooperative_matrix_KHR`
- **Validation layer**: optional `--features validation` enables `VK_LAYER_KHRONOS_validation` injection

## Observability & Diagnostics

- [ ] `tracing` span on every `VulkanBackend::*` entry point with kernel + dispatch dims
- [ ] `VK_EXT_debug_utils` integration for named objects (shader modules, pipelines) for RenderDoc / Nsight Graphics
- [ ] Validation-layer error capture funneled to `VulkanError::ValidationFailed` rather than swallowed
- [ ] Pipeline-cache hit / miss statistics counters
- [ ] `VK_KHR_performance_query` gated on hardware support -- expose kernel-level GPU timestamps

## Roadmap & Milestones

- **v0.2 (Tensor-Core parity)**: `VK_KHR_cooperative_matrix` FP16/BF16/FP8 GEMM, subgroup-size-control, push descriptors
- **v0.3 (Async & Memory)**: timeline semaphore chains, VMA-style sub-allocator, atomic float reductions
- **v0.4 (Polish)**: Validation-layer-clean across all vendors, SPIR-V 1.6 conditional path, fuzz harness
- **v1.0 (Stable)**: Vendor-native performance parity (≥ 90%) on NVIDIA / AMD / Intel for representative workloads
