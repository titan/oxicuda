# oxicuda-backend TODO

Abstract `ComputeBackend` trait that lets higher-level crates (SciRS2, oxionnx, ToRSh, TrustformeRS) dispatch GPU work without coupling to any specific GPU API. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, zero external dependencies.

## Implementation Status

**Actual: 704 SLoC across 1 file** (`src/lib.rs`)

The crate is intentionally tiny: it exposes only the trait, supporting enums, and an error type. Concrete implementations live in their own crates (`oxicuda` for CUDA, `oxicuda-rocm`, `oxicuda-metal`, `oxicuda-vulkan`, `oxicuda-webgpu`, `oxicuda-levelzero`). The trait is object-safe (`Box<dyn ComputeBackend>`) and `Send + Sync`, so consumers can hold a runtime-selected backend behind dynamic dispatch.

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

#### `ComputeBackend` trait (object-safe, `Send + Sync + Debug`)
- [x] `name() -> &str` -- backend identifier (`"cuda"`, `"rocm"`, ...)
- [x] `init(&mut self) -> BackendResult<()>` -- pick device + create context (idempotent)
- [x] `is_initialized(&self) -> bool`
- [x] `gemm(...)` -- `C = alpha * op(A) * op(B) + beta * C` (column-major f64, with leading dimensions and transpose modes)
- [x] `conv2d_forward(...)` -- NCHW 2D convolution with stride and padding
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
- [ ] `gemm_mixed_precision(...)` -- explicit FP16/BF16/FP8 input + FP32 accumulate signatures so backends can hit Tensor Cores / WMMA paths without inferring from buffer types
- [ ] `conv2d_backward_data` / `conv2d_backward_filter` -- gradient passes (currently only forward is in the trait)
- [ ] `softmax(axis, input, output, shape)` -- numerically-stable softmax as a first-class op (today consumers must compose `reduce(Max) + unary(Exp) + reduce(Sum) + binary(Div)`)
- [ ] `gather` / `scatter` / `index_select` -- indexed memory operations needed by embedding tables and MoE routing

#### P1 -- Capability discovery
- [ ] `Capabilities` query -- per-backend struct reporting `supports_fp16`, `supports_bf16`, `supports_fp8`, `tensor_cores`, `peer_access`, `unified_memory`, `max_threads_per_block`, `max_shared_mem_per_block`
- [ ] `recommended_tile_for(shape)` -- backend-specific GEMM tile-shape hint exposed to autotuners
- [ ] `Device` enumeration trait (`available_devices() -> Vec<DeviceInfo>`) returning name, total memory, compute capability / shader model

#### P2 -- Asynchronous execution
- [ ] `Stream` opaque handle in the trait (currently every op implicitly uses the default stream)
- [ ] `Event` opaque handle for cross-stream dependencies
- [ ] Async variants `copy_htod_async(stream, ...)`, `copy_dtoh_async(stream, ...)`, `gemm_async(stream, ...)`
- [ ] `record_event` / `wait_event` for graph-level scheduling

#### P2 -- Quality of life
- [ ] Blanket `impl<T: ComputeBackend + ?Sized> ComputeBackend for &mut T` so consumers can pass `&mut dyn ComputeBackend` without re-boxing
- [ ] A `NullBackend` that returns `Unsupported` for every op -- useful as a test scaffold so consumers can compile without pulling a real GPU backend

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none)     | Trait uses only `std` types (`std::fmt`, `std::error::Error`) | Yes |

Zero external dependencies on purpose -- this crate is the abstract seam between consumers and concrete GPU backends, so it must not pull in anything that any backend would also pull.

## Quality Status

- Warnings: 0
- Tests: 7 unit tests (trait shape + default `batched_gemm` behaviour)
- unwrap() calls: 0
- clippy: clean (pedantic + nursery); `#[allow(clippy::too_many_arguments)]` on `gemm` / `conv2d_forward` / `attention` / `batched_gemm` is intentional given the BLAS-style API

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
- [ ] Add a `cp_async_copy` trait method for global → shared async copies (today this happens inside concrete backends)
- [ ] Surface a structured `wmma`/`mma.sync` tile descriptor in the trait so consumers can hand-pick MMA shapes

### Hopper (sm_90) / Blackwell (sm_100)
- [ ] Expose a Tensor Memory Accelerator (TMA) descriptor opaque type for backends that can use `cp.async.bulk`
- [ ] Cluster-launch (thread-block clusters) capability flag in `Capabilities`

---

## Deepening Opportunities

> Items marked `[x]` represent trait surface coverage. The items below represent the gap between the present abstract surface and the full feature set offered by NVIDIA's cuBLAS/cuDNN/cuFFT.

### Test Coverage Gaps
- [ ] Cross-backend conformance test suite (same input → same numerical result within tolerance across `cuda`, `rocm`, `metal`, `vulkan`, `webgpu`, `levelzero`)
- [ ] Property-based tests for `BackendTranspose` × `gemm` -- verify `op(A) * op(B)` matches a reference CPU implementation across all 9 transpose combinations
- [ ] Stress test that `Box<dyn ComputeBackend>` is genuinely object-safe by holding one in a `Vec` of mixed backends

### Implementation Deepening
- [ ] A `MockBackend` that records every call (operation kind, arguments, byte counts) so consumer crates can unit-test their dispatch logic without a GPU
- [ ] Doc-tests demonstrating each trait method against `MockBackend` to keep the documented contract executable
- [ ] An async helper crate (`oxicuda-backend-async`) that wraps `ComputeBackend` with `tokio` channels for callers that want futures rather than blocking calls
