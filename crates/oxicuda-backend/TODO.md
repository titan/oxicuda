# oxicuda-backend TODO

Abstract `ComputeBackend` trait that lets higher-level crates (SciRS2, oxionnx, ToRSh, TrustformeRS) dispatch GPU work without coupling to any specific GPU API. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, zero external dependencies.

## Implementation Status

**Trait + host-side control plane across 8 files** (`src/lib.rs`, `error.rs`, `ops.rs`, `capabilities.rs`, `backend_kind.rs`, `registry.rs`, `cpu.rs`, `null.rs`), still **zero external dependencies**.

The crate exposes the object-safe `ComputeBackend` trait (`Box<dyn ComputeBackend>`, `Send + Sync`) plus the multi-backend abstraction layer that makes runtime-selected dispatch usable: `BackendKind`, `BackendRegistry` (capability-based selection + GPU→CPU fallback chains + op routing), `Capabilities`/`DeviceInfo` discovery, a genuinely-working pure-Rust `CpuBackend` reference/fallback, and a `NullBackend` test scaffold. GPU execution still lives in the concrete crates (`oxicuda` for CUDA, `oxicuda-rocm`, `oxicuda-metal`, `oxicuda-vulkan`, `oxicuda-webgpu`, `oxicuda-levelzero`).

### Completed [x]

#### Error types
- [x] `BackendError` -- 5 variants: `Unsupported(String)`, `DeviceError(String)`, `InvalidArgument(String)`, `OutOfMemory`, `NotInitialized`
- [x] `Display` + `std::error::Error` impls for `BackendError`
- [x] `BackendResult<T> = Result<T, BackendError>` alias

#### Operation enums
- [x] `BackendTranspose` -- `NoTrans` (`"N"`), `Trans` (`"T"`), `ConjTrans` (`"C"`); `Clone + Copy + PartialEq + Eq + Hash + Display`
- [x] `ReduceOp` -- `Sum`, `Max`, `Min`, `Mean`; full enum traits + lowercase `Display`
- [x] `UnaryOp` -- `Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt`, `Abs`, `Neg` (8 variants)
- [x] `BinaryOp` -- `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min` (6 variants)
- [x] `MixedPrecision` -- `F16`, `Bf16` reduced-precision **input storage** selector for `gemm_mixed_precision`; full enum traits + lowercase `Display` (`src/ops.rs`)

#### `ComputeBackend` trait (object-safe, `Send + Sync + Debug`)
- [x] `name() -> &str` -- backend identifier (`"cuda"`, `"rocm"`, ...)
- [x] `init(&mut self) -> BackendResult<()>` -- pick device + create context (idempotent)
- [x] `is_initialized(&self) -> bool`
- [x] `gemm(...)` -- `C = alpha * op(A) * op(B) + beta * C` (column-major f64, with leading dimensions and transpose modes)
- [x] `gemm_mixed_precision(prec, ...)` -- col-major f32 GEMM with f16/bf16 input storage and f32 accumulate (default `Unsupported`; real host reference in `src/cpu.rs`)
- [x] `conv2d_forward(...)` -- NCHW 2D convolution with stride and padding
- [x] `conv2d_backward_data(...)` / `conv2d_backward_filter(...)` -- conv2d gradient passes (default `Unsupported`; real host reference in `src/cpu.rs`, verified by finite-difference gradient check)
- [x] `attention(...)` -- scaled dot-product attention with optional causal masking
- [x] `reduce(op, input, output, shape, axis)` -- axis-wise reduction
- [x] `unary(op, input, output, n)` -- element-wise unary
- [x] `binary(op, a, b, output, n)` -- element-wise binary
- [x] `batched_gemm(...)` -- default implementation loops over `gemm` with byte offsets; concrete backends should override with a single batched kernel
- [x] `synchronize()` -- block host until all submitted GPU work completes
- [x] `alloc(bytes) -> BackendResult<u64>` / `free(ptr)` -- opaque device pointers
- [x] `copy_htod(dst, src: &[u8])` / `copy_dtoh(dst: &mut [u8], src)` -- byte-slice host/device transfers

#### Unit tests (7 tests)
- [x] `backend_error_display` -- all 5 error variants format correctly
- [x] `backend_error_is_std_error` -- boxed `dyn std::error::Error` round-trip
- [x] `backend_transpose_display_and_values` -- single-letter display and equality
- [x] `reduce_op_display_and_coverage` -- all 4 ops display + iterate
- [x] `unary_op_display_and_coverage` -- all 8 ops display + iterate
- [x] `binary_op_display_and_coverage` -- all 6 ops display + iterate
- [x] `enum_clone_and_hash` -- enum participation in `HashSet`, `Copy`, `Clone`
- [x] `MockBackend` + `batched_gemm_{zero_batch_is_noop,default_calls_gemm_n_times,single_batch}` -- verifies the default implementation issues exactly `batch_count` `gemm` calls and is a no-op for `batch_count == 0`

### Future Enhancements [ ]

#### P0 -- Trait surface widening (driven by consumer crates)
- [x] `gemm_mixed_precision(...)` -- explicit FP16/BF16 input + FP32 accumulate signature so backends can hit Tensor Cores / WMMA paths without inferring from buffer types (default trait method in `src/lib.rs:ComputeBackend::gemm_mixed_precision`; host reference in `src/cpu.rs:CpuBackend::gemm_mixed_precision` rounds the col-major f32 operands to the chosen 16-bit storage via `src/precision.rs`, then accumulates in f32). Storage-format selector `MixedPrecision::{F16,Bf16}` in `src/ops.rs`. IEEE-754 binary16/bfloat16 round-to-nearest-even (incl. subnormals, overflow saturation, carry-into-exponent) in `src/precision.rs:{round_to_f16,round_to_bf16}`. FP8 (E4M3/E5M2) left `[ ]` -- *(no consumer needs it yet; add a third `MixedPrecision` variant + encoder when one does)*. GPU backends still need a Tensor-Core kernel -- *(requires GPU hardware)*
- [x] `conv2d_backward_data` / `conv2d_backward_filter` -- gradient passes (forward-only before) (default trait methods in `src/lib.rs`; host reference in `src/cpu.rs:CpuBackend::{conv2d_backward_data,conv2d_backward_filter}`. `backward_data` = full convolution of the upstream gradient with the flipped filter; `backward_filter` = correlation of input with the upstream gradient. NCHW/KCHW f32, matching `conv2d_forward`, with the same stride/padding and output-extent validation). GPU backends still need fused kernels -- *(requires GPU hardware)*
- [x] `softmax(axis, input, output, shape)` -- numerically-stable softmax as a first-class op (default trait method in `src/lib.rs`; really implemented on the host by `CpuBackend` in `src/cpu.rs`. GPU backends still need a fused kernel -- *(requires GPU hardware)*)
- [x] `gather` / `scatter` -- indexed memory operations needed by embedding tables and MoE routing (default trait methods in `src/lib.rs`; host implementation in `src/cpu.rs`). `index_select` is `gather` with a 1-D index list.

#### P1 -- Capability discovery
- [x] `Capabilities` query -- per-backend struct reporting `supports_fp16`, `supports_bf16`, `supports_fp8`, `tensor_cores`, `peer_access`, `unified_memory`, `max_threads_per_block`, `max_shared_mem_per_block` (+ `cluster_launch`, `async_copy`, `warp_size`) in `src/capabilities.rs`; exposed via `ComputeBackend::capabilities()` (`src/lib.rs`)
- [x] `recommended_tile_for(shape)` -- GEMM tile-shape hint exposed to autotuners (`TileShape` + `default_tile_for` in `src/capabilities.rs`; `ComputeBackend::recommended_tile_for()` default method in `src/lib.rs`, snaps to WMMA-aligned tiles when Tensor Cores are reported)
- [x] `Device` enumeration (`available_devices() -> Vec<DeviceInfo>`) returning name, total memory, compute capability / shader model, memory kind, capabilities (`DeviceInfo`/`MemoryKind` in `src/capabilities.rs`; `ComputeBackend::available_devices()` default method; `CpuBackend` reports a single host device). *Populating it for a real GPU requires the device driver -- (requires GPU hardware)*

#### P2 -- Asynchronous execution  *(requires GPU hardware -- real streams/events need a device runtime)*
- [ ] `Stream` opaque handle in the trait (currently every op implicitly uses the default stream) *(requires GPU hardware)*
- [ ] `Event` opaque handle for cross-stream dependencies *(requires GPU hardware)*
- [ ] Async variants `copy_htod_async(stream, ...)`, `copy_dtoh_async(stream, ...)`, `gemm_async(stream, ...)` *(requires GPU hardware)*
- [ ] `record_event` / `wait_event` for graph-level scheduling *(requires GPU hardware)*

#### P2 -- Quality of life
- [x] Blanket `impl<T: ComputeBackend + ?Sized> ComputeBackend for &mut T` so consumers can pass `&mut dyn ComputeBackend` without re-boxing (`src/lib.rs`)
- [x] A `NullBackend` that returns `Unsupported` for every op -- useful as a test scaffold so consumers can compile without pulling a real GPU backend (`src/null.rs`)

#### NEW -- Multi-backend control plane (host-side, no GPU needed)
- [x] `BackendKind` enum (`Cuda`/`Rocm`/`LevelZero`/`Vulkan`/`Metal`/`WebGpu`/`Cpu`) with default preference ordering, `is_gpu`, name parse/aliases (`src/backend_kind.rs`)
- [x] `BackendRegistry` -- register/replace backends, runtime availability flags, capability-based `select`/`select_best`, GPU→CPU `fallback_chain` (CPU forced last), `OpClass` routing that prefers Tensor-Core backends for MatMul/Conv/Attention (`src/registry.rs`)
- [x] `SelectionRequest` feature negotiation (`require_gpu`/fp16/bf16/fp8/tensor_cores/unified_memory/peer_access/`pin`) with honest "no backend available" errors (`src/registry.rs`)
- [x] `CpuBackend` -- genuinely-working pure-Rust reference backend with real host memory table (never reuses freed pointers), real gemm/gemm_mixed_precision/conv2d_forward/conv2d_backward_data/conv2d_backward_filter/attention/reduce/unary/binary/softmax/gather/scatter (`src/cpu.rs`). Serves as the always-available fallback **and** the numerical reference for conformance.

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none)     | Trait uses only `std` types (`std::fmt`, `std::error::Error`) | Yes |

Zero external dependencies on purpose -- this crate is the abstract seam between consumers and concrete GPU backends, so it must not pull in anything that any backend would also pull.

## Quality Status

- Warnings: 0 (rustc + clippy `-D warnings`, all-targets, all-features; rustdoc clean)
- Tests: 101 unit + conformance tests across `error`/`ops`/`capabilities`/`backend_kind`/`registry`/`cpu`/`null`/`precision`/trait-defaults/conformance modules (was 77). New: 14 `precision` IEEE-754 round-trip/edge-case tests, 6 mixed-precision GEMM tests (bf16/f16 vs f32 tolerance, exact-for-representable, **f32-accumulation-not-f16** proof, alpha/beta+transpose, bad-ld), 4 conv-backward tests (finite-difference gradient checks for data+filter, known 1x1, shape-mismatch rejection)
- unwrap() calls: 0 in non-test code
- clippy: clean; `#[allow(clippy::too_many_arguments)]` on `gemm` / `gemm_mixed_precision` / `conv2d_forward` / `conv2d_backward_data` / `conv2d_backward_filter` / `attention` / `batched_gemm` is intentional given the BLAS/cuDNN-style API
- Files: `src/lib.rs` (trait + blanket impl + conformance), `src/error.rs`, `src/ops.rs`, `src/capabilities.rs`, `src/backend_kind.rs`, `src/registry.rs`, `src/cpu.rs`, `src/null.rs`, `src/precision.rs` (pure-Rust f16/bf16 round-to-nearest-even storage emulation) -- each well under the 2000-line limit (cpu.rs 1875), zero external dependencies retained

## Performance Targets

The trait crate itself has no runtime cost: every method is a virtual call through a vtable. The relevant performance characteristics are owned by concrete backends:

| Backend | Crate | Status |
|---------|-------|--------|
| CUDA | `oxicuda` (top-level) | Full coverage, dispatches to `oxicuda-blas`, `oxicuda-dnn` |
| ROCm/HIP | `oxicuda-rocm` | 56 tests, all 7 compute ops wired |
| Metal | `oxicuda-metal` | 121 tests, MSL shader dispatch |
| Vulkan | `oxicuda-vulkan` | 66 tests, SPIR-V compute shaders |
| WebGPU | `oxicuda-webgpu` | 86 tests, WGSL shader dispatch |
| Level Zero | `oxicuda-levelzero` | 69 tests, OpenCL SPIR-V kernels |

## Notes

- The trait uses raw `u64` device pointers (matching `CUdeviceptr`) so the same trait works for backends whose native pointer type is a Vulkan `VkBuffer` ID, a Metal `MTLBuffer`, or a WebGPU `GPUBuffer` handle, all of which can round-trip through `u64`.
- All memory traffic flows through `&[u8]` / `&mut [u8]` (rather than `&[T]` for some `T: Pod`) to keep the trait object-safe.
- `batched_gemm` has a default implementation so backends only need to override it if they have a dedicated batched kernel.

---

## Architecture-Specific Deepening Opportunities

### Ampere (sm_80) / Ada (sm_89)
- [ ] Add a `cp_async_copy` trait method for global → shared async copies (today this happens inside concrete backends) *(requires GPU hardware -- `Capabilities::async_copy` flag is wired in `src/capabilities.rs`, but the actual `cp.async` op needs a device)*
- [ ] Surface a structured `wmma`/`mma.sync` tile descriptor in the trait so consumers can hand-pick MMA shapes *(requires GPU hardware -- WMMA-aligned tile hints already returned by `recommended_tile_for`; a real MMA op descriptor needs a device)*

### Hopper (sm_90) / Blackwell (sm_100)
- [ ] Expose a Tensor Memory Accelerator (TMA) descriptor opaque type for backends that can use `cp.async.bulk` *(requires GPU hardware)*
- [x] Cluster-launch (thread-block clusters) capability flag in `Capabilities` (`cluster_launch` field in `src/capabilities.rs`). *Acting on it requires GPU hardware.*

---

## Deepening Opportunities

> Items marked `[x]` represent trait surface coverage. The items below represent the gap between the present abstract surface and the full feature set offered by NVIDIA's cuBLAS/cuDNN/cuFFT.

### Test Coverage Gaps
- [~] Cross-backend conformance test suite. **CPU-reference half done**: `mod conformance` in `src/lib.rs` pins every op's documented contract against `CpuBackend`. The same property checks run against `cuda`/`rocm`/`metal`/`vulkan`/`webgpu`/`levelzero` output *(requires GPU hardware)* and live in the concrete backend crates.
- [x] Property-based tests for `BackendTranspose` × `gemm` -- `gemm_matches_reference_across_all_transpose_combos` (`src/lib.rs`) verifies all 9 transpose combinations against a naive reference GEMM on random inputs (deterministic LCG, full ÷(2³²-1) range).
- [x] Stress test that `Box<dyn ComputeBackend>` is genuinely object-safe by holding one in a `Vec` of mixed backends -- `object_safety_vec_of_mixed_backends` (`src/lib.rs`) holds `MockBackend`/`CpuBackend`/`NullBackend` together.

### Implementation Deepening
- [x] A `MockBackend` that records every call (operation kind + byte/elem counts) so consumer crates can unit-test their dispatch logic without a GPU (`mod tests` in `src/lib.rs`; `mock_records_dispatch_for_consumer_tests`).
- [ ] Doc-tests demonstrating each trait method against `MockBackend` -- deferred (the `MockBackend` lives in `#[cfg(test)]`; making it public for doc-tests is a separate API decision).
- [ ] An async helper crate (`oxicuda-backend-async`) that wraps `ComputeBackend` with `tokio` channels -- **out of scope here**: it is a *separate crate* and would add an external `tokio` dependency, which this zero-dependency crate must not pull.
