# OxiCUDA TODO

Pure Rust CUDA replacement for the COOLJAPAN ecosystem.
(C) 2026 COOLJAPAN OU (Team KitaSan)

## Project Status (v0.1.5 — 2026-05-03)

- **Crates**: 37 workspace members (36 library crates + 1 umbrella)
- **Files**: 1000+ Rust source files
- **Code**: ~310,000 SLoC (Rust)
- **Tests**: 9,568 passing (workspace-wide), 2 skipped (GPU-only on macOS)
- **Warnings**: 0 (clippy + rustc, `-D warnings`)
- **unwrap() calls**: 0 (no-unwrap policy in library code)
- **Status**: Vol.1–25 complete — Vol.1 Foundation, Vol.2 PTX/Autotune, Vol.3 BLAS, Vol.4 DNN, Vol.5 Scientific, Vol.6 Signal, Vol.7 Computation Graph, Vol.8 Training, Vol.9 Inference, Vol.10 RL, Vol.11 High-Perf Inference, Vol.12 Distributed, Vol.13 LLM Primitives, Vol.14–16 backend crates, **Vol.17 Generative AI**, **Vol.18 Graph Neural Networks**, **Vol.19 State Space Models (Mamba/S4/RWKV)**, **Vol.20 Vision Transformers & CLIP**, **Vol.21 Audio/Speech ML (Conformer/Wav2Vec2/CTC/WaveNet/SpecAugment/x-vector)**, **Vol.22 Time-Series Forecasting (TCN/NHiTS/PatchTST/TimesNet/iTransformer/RevIN)**, **Vol.23 Bayesian Deep Learning**, **Vol.24 Federated Learning**, **Vol.25 Neural Architecture Search**; SDE samplers added to oxicuda-rand

## Design Principles

- **Pure Rust**: Zero C/Fortran dependencies. No CUDA SDK, no nvcc, no C toolchain.
- **Minimal Dependencies**: Only `libloading`, `thiserror`, `num-complex`, `half` (optional), `serde` (optional).
- **Runtime-only GPU**: `libcuda.so` / `nvcuda.dll` loaded dynamically at runtime via `libloading`.
- **No Unwrap**: All fallible operations return `Result<T, E>`. No `unwrap()` or `expect()` in library code.
- **No Warnings**: Zero clippy warnings, zero compiler warnings across the entire workspace.
- **Workspace Inheritance**: Single version source of truth via `Cargo.toml` workspace.

See [oxicuda-estimation.md](oxicuda-estimation.md) for detailed project estimation
(original estimate: 1.6M-3.1M SLoC; actual: 42K SLoC core implementation).

---

## Vol.1: Foundation Architecture & Driver Layer [COMPLETE]

### oxicuda-driver (24 files, 8,970 SLoC)
- [x] FFI type definitions (ffi.rs) -- CUdevice, CUcontext, CUstream, CUmodule, CUfunction
- [x] Error handling (error.rs) -- CudaError with ~100 CUDA result code variants
- [x] Dynamic loader (loader.rs) -- libcuda.so / nvcuda.dll via libloading, no build-time SDK
- [x] Device management (device.rs) -- enumeration, attributes, compute capability queries
- [x] Context management (context.rs) -- RAII CudaContext with push/pop stack semantics
- [x] Stream management (stream.rs) -- async streams, synchronization, callback support
- [x] Event management (event.rs) -- timing, inter-stream synchronization, elapsed time
- [x] Module/PTX loading (module.rs) -- load PTX/cubin, function handle extraction
- [x] Occupancy API (occupancy.rs) -- max active blocks, suggested block size
- [x] Multi-GPU context management (multi_gpu.rs) -- DevicePool with per-device context pool, round-robin scheduling, best_available_device selection
- [x] Graph API (graph.rs) -- Graph, GraphNode, GraphExec, StreamCapture (cudaGraph equivalent)
- [x] Driver version queries (device.rs) -- cuDriverGetVersion
- [x] Peer-to-peer access (device.rs) -- cuDeviceCanAccessPeer
- [x] Primary context management (primary_context.rs) -- cuDevicePrimaryCtxRetain/Release

### oxicuda-memory (17 files, 4,081 SLoC)
- [x] DeviceBuffer<T> (device_buffer.rs) -- typed GPU allocation, Send + Sync, RAII free
- [x] PinnedBuffer<T> (host_buffer.rs) -- page-locked host memory for async transfers
- [x] Copy operations (copy.rs) -- H2D, D2H, D2D, async variants with stream ordering
- [x] Unified memory (unified.rs) -- managed memory with automatic page migration
- [x] Memory pool (pool.rs) -- feature-gated async memory pool (CUDA 11.2+)
- [x] Zero-copy memory (zero_copy.rs) -- host-mapped device-accessible memory
- [x] Async copy operations (copy.rs) -- copy_htod_async_raw, copy_dtoh_async_raw, copy_dtod_async with stream ordering
- [x] Multi-GPU peer copy (peer_copy.rs) -- can_access_peer, enable/disable_peer_access, copy_peer, copy_peer_async
- [x] Async memory pool enhancements (pool.rs) -- PoolStats (allocated/peak/count), trim(), set_threshold()
- [x] Virtual memory management (virtual_memory.rs) -- VirtualAddressRange, PhysicalAllocation, VirtualMemoryManager
- [x] 2D/3D memory copy (copy_2d3d.rs) -- Memcpy2DParams, Memcpy3DParams, copy_2d/3d functions
- [x] Memory advice / prefetch hints (memory_info.rs) -- cuMemAdvise, cuMemPrefetchAsync
- [x] Buffer views and reinterpret cast (buffer_view.rs) -- type-safe buffer reinterpretation
- [x] Memory usage query (memory_info.rs) -- cuMemGetInfo (free/total VRAM)
  - [x] Memory pool statistics (pool_stats.rs) -- AllocationHistogram, FragmentationMetrics, PoolStatsTracker
  - [x] Managed memory hints (managed_hints.rs) -- ManagedMemoryHints, MigrationPolicy, PrefetchPlan

### oxicuda-launch (15 files, 4,161 SLoC)
- [x] Dim3 + grid helpers (grid.rs) -- 1D/2D/3D dimension types, grid size calculators
- [x] LaunchParams builder (params.rs) -- type-safe kernel launch configuration
- [x] Kernel + KernelArgs trait (kernel.rs) -- compile-time argument type validation
- [x] launch! macro (macros.rs) -- CUDA-style <<<grid, block, smem, stream>>> syntax
- [x] Launch bounds validation (error.rs, params.rs) -- LaunchParams::validate() checks block/grid/shared memory against device limits
- [x] Occupancy-based auto grid sizing (grid.rs) -- auto_grid_for(), auto_grid_2d() using occupancy API
- [x] Cooperative launch (cooperative.rs) -- CooperativeLaunch with max_active_blocks, optimal_block_size
- [x] Graph-based launch (graph_launch.rs) -- GraphLaunchCapture for launch recording
- [x] Cluster launch (cluster.rs) -- ClusterDim, ClusterLaunchParams (Hopper+)
- [x] Multi-stream launch (multi_stream.rs) -- multi_stream_launch across streams
- [x] Named kernel arguments (named_args.rs) -- NamedKernelArgs trait, ArgBuilder

---

## Vol.2: PTX Code Generator & Autotuner Engine [COMPLETE]

### oxicuda-ptx (48 files, 24,828 SLoC)
- [x] PTX IR type system (ir/types.rs) -- .f16, .bf16, .f32, .f64, .b8/.b16/.b32/.b64, .pred
- [x] Register allocator (ir/register.rs) -- virtual register allocation, spill tracking
- [x] Instruction representation (ir/instruction.rs) -- typed PTX instruction encoding
- [x] Operand types (ir/operand.rs) -- register, immediate, address, predicate operands
- [x] Basic block (ir/block.rs) -- labeled blocks with instruction sequences
- [x] Function definition (ir/function.rs) -- kernel/device function with parameters
- [x] Module definition (ir/module.rs) -- .version, .target, global declarations
- [x] KernelBuilder DSL (builder/kernel_builder.rs) -- fluent API for kernel construction
- [x] BodyBuilder (builder/body_builder.rs) -- instruction-level builder within kernel body
- [x] Architecture rules (arch.rs) -- SM 7.5 through SM 10.0 capability tables
- [x] Tensor Core generation -- WMMA (tensor_core/wmma.rs), MMA (tensor_core/mma.rs), WGMMA (tensor_core/wgmma.rs)
- [x] PTX text emitter (emit/printer.rs) -- IR to PTX text serialization
- [x] PTX validator (emit/validator.rs) -- structural and semantic validation
- [x] Disk-based PTX cache (cache.rs) -- hash-keyed file cache for compiled PTX
- [x] Atomic operations (ir/instruction.rs) -- Atom, AtomCas, Red instructions; AtomOp enum (Add, Min, Max, Inc, Dec, And, Or, Xor, Exch); 20 BodyBuilder methods
- [x] Bit manipulation (ir/instruction.rs) -- Brev, Clz, Popc, Bfind, Bfe, Bfi instructions; 21+ BodyBuilder methods
- [x] Special math (ir/instruction.rs) -- Rcp, Rsqrt, Sqrt, Ex2, Lg2, Sin, Cos instructions with rounding modes
- [x] Register pressure analysis (analysis/register_pressure.rs) -- peak tracking, spill risk, occupancy estimation
- [x] Dead code elimination (analysis/dead_code.rs) -- fixed-point DCE with liveness analysis
- [x] GEMM template (templates/gemm.rs) -- parameterized GEMM kernel generation
- [x] Elementwise template (templates/elementwise.rs) -- unary/binary elementwise ops
- [x] Reduction template (templates/reduction.rs) -- parallel reduction kernels
- [x] Softmax template (templates/softmax.rs) -- numerically stable softmax kernel
- [x] Scan/prefix-sum template (templates/scan.rs) -- Blelloch work-efficient scan with inclusive/exclusive, sum/product/min/max ops
- [x] Transpose template (templates/transpose.rs) -- coalesced shared-memory transpose with bank-conflict-free padding
- [x] Attention template (templates/attention.rs) -- FlashAttention-style fused attention kernel
- [x] Batch normalization template (templates/batch_norm.rs) -- training + inference BN kernels
- [x] MoE template (templates/moe.rs) -- top-k gating, permute, expert GEMM, unpermute
- [x] Convolution template (templates/convolution.rs) -- im2col, direct conv, 1x1 optimized, backward data/filter
- [x] Video instructions: dp4a, dp2a (ir/instruction.rs, builder/body_builder.rs)
- [x] PTX-level loop unrolling -- pragma unroll, manual unroll in builder
- [x] Integer multiply-add (ir/instruction.rs, builder/body_builder.rs) -- mad.lo, mad.hi, mad.wide instructions
- [x] Texture/surface instructions (ir/instruction.rs, ir/texture.rs, builder/body_builder.rs) -- tex.1d/2d/3d, suld, sust with 5 analysis passes updated
- [x] Constant folding optimization pass (analysis/constant_folding.rs) -- simplify constant expressions at IR level
- [x] Strength reduction optimization pass (analysis/strength_reduction.rs) -- replace expensive ops with cheaper equivalents

### oxicuda-autotune (28 files, 13,039 SLoC)
- [x] Search space definition (search_space.rs) -- parameterized kernel variant spaces
- [x] Benchmark engine (benchmark.rs) -- GPU timing with warmup, statistical analysis
- [x] TunableKernel trait (tunable.rs) -- interface for autotunable kernel implementations
- [x] Configuration types (config.rs) -- tile sizes, vector widths, unroll factors
- [x] Result database (result_db.rs) -- JSON-backed per-GPU tuning result persistence
- [x] Runtime dispatcher (dispatch.rs) -- 3-tier fallback (cached > tuned > default)
- [x] Early stopping (early_stopping.rs) -- EarlyStoppingConfig, EarlyStoppingTracker, patience-based/time-budget/convergence detection
- [x] Bayesian optimization search (bayesian.rs) -- GP surrogate + acquisition functions (EI, UCB, PI)
- [x] Simulated annealing search (simulated_annealing.rs) -- temperature-based exploration for large search spaces
- [x] Genetic algorithm search (genetic.rs) -- crossover/mutation on config populations
- [x] PTX template integration (ptx_integration.rs) -- direct SearchSpace generation from template parameters
- [x] Problem size interpolation (interpolation.rs) -- nearest-neighbor and inverse-distance-weighted interpolation
- [x] Error types (error.rs) -- autotune-specific error handling

---

## Vol.3: Linear Algebra Primitives -- cuBLAS equivalent [COMPLETE]

### oxicuda-blas (72 files, 19,913 SLoC)
- [x] GpuFloat trait hierarchy (types.rs) -- F16, BF16, TF32, F32, F64, FP8
- [x] BlasHandle (handle.rs) -- session handle with stream and workspace binding
- [x] Error types (error.rs) -- BLAS-specific error variants
- [x] BLAS Level 1 -- vector-vector operations
  - [x] axpy (y = alpha * x + y)
  - [x] scal (x = alpha * x)
  - [x] dot (dot product)
  - [x] nrm2 (L2 norm)
  - [x] asum (L1 absolute sum)
  - [x] iamax (index of max absolute value)
  - [x] copy_vec (vector copy)
  - [x] swap (vector swap)
- [x] BLAS Level 2 -- matrix-vector operations
  - [x] gemv (y = alpha * A * x + beta * y)
  - [x] symv (symmetric matrix-vector multiply)
  - [x] trmv (triangular matrix-vector multiply)
  - [x] trsv (triangular solve: T * x = b)
  - [x] ger (rank-1 update: A += alpha * x * y^T)
  - [x] syr (symmetric rank-1 update)
- [x] BLAS Level 3 -- matrix-matrix operations
  - [x] gemm (general matrix multiply with dispatch)
    - [x] SIMT path (simt.rs) -- CUDA Core non-Tensor-Core path
    - [x] Tensor Core path (tensor_core.rs) -- WMMA/MMA dispatch
    - [x] Split-K parallelization (splitk.rs) -- for tall-skinny matrices
    - [x] Epilogue fusion (epilogue.rs) -- D = alpha*A@B + beta*C + bias + activation
    - [x] Dispatch logic (dispatch.rs) -- precision x arch optimal kernel selection
  - [x] gemm_api (gemm_api.rs) -- high-level GEMM entry point
  - [x] symm (symmetric matrix multiply)
  - [x] trsm (triangular solve: T * X = B)
  - [x] syrk (C = alpha * A * A^T + beta * C)
  - [x] syr2k (C = alpha * (A*B^T + B*A^T) + beta * C)
  - [x] trmm (triangular matrix multiply)
- [x] Batched GEMM operations
  - [x] batched_gemm (independent GEMM batch execution)
  - [x] strided_gemm (strided batched GEMM)
  - [x] grouped_gemm (variable-size GEMM groups)
- [x] Precision-specific optimizations
  - [x] f64_ops (FP64 DGEMM for scientific computing)
  - [x] f32_ops (FP32 SGEMM)
  - [x] f16_ops (FP16 HGEMM with Tensor Core)
  - [x] bf16_ops (BF16 GEMM with Tensor Core)
  - [x] tf32_ops (TF32 GEMM, Ampere+)
  - [x] fp8_ops (FP8 GEMM, Hopper+)
  - [x] mixed (mixed-precision accumulation)
  - [x] int_ops (INT4/INT8 GEMM for inference via dp4a-accelerated INT8 + packed INT4)
- [x] Elementwise operations
  - [x] unary (relu, gelu, sigmoid, silu, tanh, exp, log, sqrt, abs, neg)
  - [x] binary (add, sub, mul, div, max, min, pow)
  - [x] ops (fused elementwise operation dispatch)
- [x] Reduction operations
  - [x] sum (parallel sum reduction)
  - [x] max / min (parallel max/min reduction)
  - [x] mean (mean with numerically stable accumulation)
  - [x] variance (Welford online variance)
  - [x] softmax (numerically stable softmax)
  - [x] ops (reduction operation dispatch)
- [x] Complex number support (complex_gemm.rs) -- CGEMM/ZGEMM, complex_gemm, complex_gemv with interleaved storage
- [x] Batched TRSM (batched_trsm.rs) -- batched triangular solve with warp/shared/blocked strategies
- [x] Stream-K GEMM (stream_k.rs) -- dynamic work partitioning across CTAs
- [x] Persistent kernel GEMM (persistent_gemm.rs) -- work-stealing via atomic counter
  - [x] Warp-specialized GEMM (gemm/warp_specialized.rs) -- producer/consumer warps overlapping global loads with Tensor Core MMA
  - [x] Non-square tile configurations (gemm/tiles.rs) -- RectangularTile, TileSelector, aspect-ratio heuristics
  - [x] FP8 GEMM dynamic tile selection (precision/fp8_ops.rs) -- shape-dependent heuristics for Fp8WorkloadClass
  - [x] SYRK/SYR2K tensor core paths (level3/syrk_tc.rs) -- triangle-masked TC kernels
  - [x] Multi-stream batched GEMM (batched/multi_stream_batched.rs) -- distribute across multiple streams

---

## Vol.4: Deep Learning Primitives -- cuDNN equivalent [COMPLETE]

### oxicuda-dnn (89 files, 31,293 SLoC)
- [x] DnnHandle (handle.rs) -- session handle for DNN operations
- [x] Error types (error.rs) -- DNN-specific error variants
- [x] Tensor utilities (tensor_util.rs) -- layout, stride, shape helpers
- [x] DNN types (types.rs) -- tensor descriptors, data formats, algorithm enums
- [x] PTX helpers (ptx_helpers.rs) -- shared PTX generation utilities
- [x] Convolution module
  - [x] Descriptors (conv/descriptor.rs) -- ConvolutionDescriptor, FilterDescriptor
  - [x] Algorithm selection (conv/algo_select.rs) -- heuristic + autotuned selection
  - [x] High-level API (conv/api.rs) -- unified convolution interface
  - [x] Forward: implicit GEMM (conv/fprop/implicit_gemm.rs)
  - [x] Forward: im2col + GEMM (conv/fprop/im2col_gemm.rs)
  - [x] Forward: Winograd 3x3 (conv/fprop/winograd.rs)
  - [x] Forward: direct 1x1 / depthwise (conv/fprop/direct.rs)
  - [x] Backward data: implicit GEMM (conv/dgrad/implicit_gemm.rs)
  - [x] Backward filter: implicit GEMM (conv/wgrad/implicit_gemm.rs)
  - [x] Fused Conv + BN + Activation (conv/fused.rs)
  - [x] Transposed convolution (conv/transpose_conv.rs) -- TransposeConvConfig, TransposeConvPlan, col2im PTX, weight reshape PTX
  - [x] 3D convolution (conv/conv3d/) -- im2col3d + GEMM, forward/backward/wgrad for volumetric data
  - [x] Depthwise separable conv fusion (conv/depthwise_separable.rs) -- fused DW+PW kernel
  - [x] Deformable convolution (conv/deformable.rs) -- DCNv2 with bilinear interpolation, forward+backward
  - [x] FFT-based convolution (conv/fft_conv.rs) -- frequency-domain convolution for large kernels (7x7+)
- [x] Attention module
  - [x] Multi-Head Attention naive (attn/mha.rs)
  - [x] FlashAttention forward (attn/flash_attn/forward.rs) -- FP16, causal mask
  - [x] FlashAttention backward (attn/flash_attn/backward.rs)
  - [x] PagedAttention (attn/flash_attn/paged.rs) -- KV-cache paging
  - [x] Decode attention (attn/flash_attn/decode.rs) -- single-query inference
  - [x] Rotary Positional Embedding (attn/rope.rs) -- RoPE for LLM position encoding
  - [x] KV-Cache management (attn/kv_cache.rs)
  - [x] Fused RoPE+attention (attn/fused_rope_attn.rs) -- single kernel RoPE + attention
  - [x] FlashAttention-3 Hopper (attn/flash_attn/hopper.rs) -- warp-specialized forward+backward, TMA, ping-pong pipeline
- [x] Mixture of Experts (MoE) module
  - [x] Top-k routing (moe/routing.rs) -- softmax-based expert selection
  - [x] Token permutation (moe/permute.rs) -- scatter/gather for expert dispatch
  - [x] Fused MoE kernel (moe/fused_moe.rs) -- single-pass MoE execution
  - [x] Grouped GEMM for MoE (moe/grouped_gemm.rs)
  - [x] MoE auxiliary loss computation (moe/aux_loss.rs) -- Switch Transformer load-balance loss + z-loss
  - [x] Expert capacity factor tuning (moe/capacity.rs) -- dynamic capacity adjustment
  - [x] MoE load balancing monitoring (moe/monitoring.rs) -- runtime utilization tracking
- [x] Normalization module
  - [x] BatchNorm forward + backward (norm/batch_norm.rs)
  - [x] LayerNorm (norm/layer_norm.rs)
  - [x] RMSNorm (norm/rms_norm.rs) -- for LLM architectures (LLaMA, etc.)
  - [x] GroupNorm (norm/group_norm.rs)
  - [x] InstanceNorm (norm/instance_norm.rs) -- per (batch, channel) normalization
  - [x] ScaleNorm (norm/scale_norm.rs) -- simplified L2 normalization
  - [x] PowerNorm (norm/power_norm.rs) -- running power mean normalization
  - [x] Fused normalization (norm/fused_norm.rs) -- norm + activation fusion
- [x] Pooling module
  - [x] MaxPool2D (pool/max_pool.rs)
  - [x] AvgPool2D (pool/avg_pool.rs)
  - [x] AdaptivePool (pool/adaptive_pool.rs)
  - [x] GlobalPool (pool/global_pool.rs)
- [x] Resize module
  - [x] Nearest-neighbor interpolation (resize/nearest.rs)
  - [x] Bilinear interpolation (resize/bilinear.rs)
  - [x] Bicubic interpolation (resize/bicubic.rs)
- [x] Quantization module
  - [x] FP8 quantization (quantize/fp8_quantize.rs) -- Hopper+
  - [x] INT8 quantization (quantize/int8_quantize.rs)
  - [x] Block-scaled FP4 (quantize/block_scale.rs) -- Blackwell
- [x] RNN module
  - [x] LSTM cell (rnn/lstm.rs) -- full forward pass for single timestep and sequence
  - [x] GRU cell (rnn/gru.rs) -- full forward pass for single timestep and sequence
- [x] INT4/NF4 quantization (quantize/int4_quantize.rs) -- quantize/dequantize with group scaling (QLoRA support)
- [x] GQA/MQA native support (attn/gqa.rs) -- grouped-query and multi-query attention
- [x] Sliding window attention (attn/sliding_window.rs) -- Mistral-style configurable window size
- [x] Fused GEMM+bias+activation epilogue (fused_linear.rs) -- fused_linear with all activations
  - [x] Winograd backward pass (conv/dgrad/winograd.rs, conv/wgrad/winograd.rs) -- backward data and filter gradients through Winograd domain
  - [x] Block-sparse attention (attn/block_sparse.rs) -- CSR-format structured sparsity patterns for long-context
  - [x] Quantization-aware training (quantize/qat.rs) -- fake quantize + straight-through estimator

---

## Vol.5.5: GPU Parallel Primitives -- CUB Equivalent [COMPLETE]

### oxicuda-primitives (16 files, ~4,200 SLoC)

CUB-equivalent parallel GPU primitives, zero CUDA SDK dependency.
All kernels are generated as PTX source strings at runtime and JIT-compiled via `cuModuleLoadData`.

- [x] Error types (error.rs) -- PrimitivesError with 9 variants, PrimitivesResult alias
- [x] Session handle (handle.rs) -- PrimitivesHandle wrapping Arc<Context> + Arc<Stream> + SmVersion
- [x] Shared PTX utilities (ptx_helpers.rs) -- ptx_header, ptx_type_str/bytes, reg_decl, ReduceOp, PrimitiveType trait
  - Full SmVersion coverage including Sm90a, Sm100, Sm120
  - Full PtxType coverage including BF16x2, F16x2, TF32, FP8 (E4M3/E5M2), FP6, FP4 (E2M1), B128
- [x] **Warp-level primitives** (warp/reduce.rs, warp/scan.rs)
  - Reduce: `shfl.sync.bfly.b32` butterfly tree; f64 split lo/hi; optional broadcast lane
  - Scan: `shfl.sync.up.b32` shift-and-combine; inclusive + exclusive; f64 lo/hi split
  - All 7 ops: Sum, Product, Min, Max, And, Or, Xor
- [x] **Block-level primitives** (block/reduce.rs, block/scan.rs)
  - Reduce: warp-reduce + shared-memory merge + `shfl.sync.bfly` for warp-level aggregation
  - Scan: work-efficient Blelloch up-sweep / down-sweep; inclusive + exclusive
  - All 7 ops; f64 split-register support throughout
- [x] **Device-wide reduce** (device/reduce.rs) -- 2-pass pipeline (partial sums → final scalar)
- [x] **Device-wide scan** (device/scan.rs) -- 3-kernel pipeline: block scan → propagate → apply
- [x] **Stream compaction** (device/select.rs) -- 2-kernel flag+gather pipeline over exclusive scan
  - `SelectPredicate`: NonZero, Positive, Negative (unsigned → always false), FlagArray
  - Type-correct `setp.{lt,gt,ne}.{ty}` per predicate × element type
- [x] **Privatized histogram** (device/histogram.rs) -- 2-kernel init+count pipeline
  - `DeviceHistogramMode`: Modulo (rem.u32/rem.u64) and EvenRange (fp or integer linear map)
  - Per-block shared-memory private histogram; `atom.shared.add.u32`; strided global merge
- [x] **4-bit LSD Radix Sort** (sort/radix_sort.rs) -- 3 kernels per pass × 8 (u32) or 16 (u64) passes
  - Count kernel: private `cnt_hist[16]` in shared memory + `atom.shared.add.u32`
  - Scan kernel: 1 block × 16 threads; sequential column scan for exclusive prefix
  - Scatter kernel: `block_offs[16]` pre-loaded + `atom.shared.add.u32` for unique output positions
- [x] **Bitonic Block Sort + Co-rank Merge Sort** (sort/merge_sort.rs)
  - Bitonic sort: 2-barrier-per-stage correctness (pre-load + pre-write); `selp.{ty}` for type-correct compare-swap
  - Merge kernel: O(log n) co-rank binary search per element; branch-based output selection
  - 154 tests: 142 unit + 12 doctests, all passing
  - ptx_helpers comprehensive coverage: all PtxType variants (B128, E4M3/E5M2/E2M3/E3M2/E2M1, F16x2, BF16x2, TF32), all SmVersion variants (Sm90a, Sm100, Sm120), all ReduceOp identities and instruction mnemonics

---

## Vol.5: Scientific Computing & Ecosystem Integration [COMPLETE]

### oxicuda-fft (35 files, 8,853 SLoC)
- [x] FftPlan (plan.rs) -- transform planning with strategy selection
- [x] FFT execution engine (execute.rs) -- plan dispatch and execution
- [x] Error types (error.rs) -- FFT-specific errors
- [x] FFT types (types.rs) -- Complex<f32>, Complex<f64>, transform direction
- [x] PTX helpers (ptx_helpers.rs) -- FFT kernel PTX generation utilities
- [x] Stockham auto-sort FFT (kernels/stockham.rs) -- GPU-optimized in-place FFT
- [x] Batched FFT (kernels/batch_fft.rs) -- multiple small FFTs in parallel
- [x] Large FFT (kernels/large_fft.rs) -- global memory multi-pass for large sizes
- [x] Matrix transpose (kernels/transpose.rs) -- for multi-dimensional FFT decomposition
- [x] Radix-2 butterfly (radix/radix2.rs)
- [x] Radix-4 butterfly (radix/radix4.rs)
- [x] Radix-8 butterfly (radix/radix8.rs)
- [x] Mixed-radix support (radix/mixed_radix.rs) -- composite sizes (2, 3, 5, 7)
- [x] Bluestein / Chirp-Z (radix/bluestein.rs) -- arbitrary-size FFT
- [x] Complex-to-Complex (transforms/c2c.rs)
- [x] Real-to-Complex (transforms/r2c.rs)
- [x] Complex-to-Real (transforms/c2r.rs)
- [x] 2D FFT (transforms/fft2d.rs)
- [x] 3D FFT (transforms/fft3d.rs)
- [x] Stockham bank conflict avoidance (bank_conflict_free.rs) -- padded layout (addr + addr/32), power-of-2 sizes 64-4096
- [x] Batched FFT kernel fusion (fused_batch.rs) -- multiple small FFTs per thread block, N<=1024, shared memory only
- [x] Split-radix FFT (split_radix.rs) -- radix-2/4 hybrid for ~10% fewer operations
- [x] Real-valued FFT optimization (real_fft.rs) -- pack/unpack exploiting conjugate symmetry
- [x] Prime-factor algorithm (pfa.rs) -- Good-Thomas FFT with CRT mapping
- [x] Convolution via FFT helper (conv_fft.rs) -- 1D/2D convolution + cross-correlation
  - [x] Multi-GPU FFT (multi_gpu.rs) -- 1D slab decomposition across P devices

### oxicuda-sparse (36 files, 11,021 SLoC)
- [x] Sparse handle (handle.rs)
- [x] Error types (error.rs)
- [x] PTX helpers (ptx_helpers.rs) -- sparse kernel PTX utilities
- [x] Storage formats
  - [x] CSR -- Compressed Sparse Row (format/csr.rs)
  - [x] CSC -- Compressed Sparse Column (format/csc.rs)
  - [x] COO -- Coordinate format (format/coo.rs)
  - [x] BSR -- Block Sparse Row (format/bsr.rs)
  - [x] ELL -- ELLPACK format (format/ell.rs)
  - [x] Format conversion (format/convert.rs) -- CSR<->CSC, COO<->CSR, etc.
- [x] Sparse operations
  - [x] SpMV -- sparse matrix-vector multiply (ops/spmv.rs)
  - [x] SpMM -- sparse matrix-matrix multiply (ops/spmm.rs)
  - [x] SpGEMM -- sparse-sparse matrix multiply (ops/spgemm.rs)
  - [x] SDDMM -- sampled dense-dense matrix multiply (ops/sddmm.rs)
  - [x] SpTRSV -- sparse triangular solve (ops/sptrsv.rs)
  - [x] Krylov subspace methods (ops/krylov.rs) -- Lanczos and Arnoldi iteration
- [x] Preconditioners
  - [x] ILU(0) -- incomplete LU factorization (preconditioner/ilu0.rs)
  - [x] IC(0) -- incomplete Cholesky factorization (preconditioner/ic0.rs)
- [x] ELL-optimized SpMV (spmv_ell.rs) -- coalesced column-major access, sentinel-based padding
- [x] BSR SpMV kernel (spmv_bsr.rs) -- block-aware SpMV, one thread block per block-row, dense sub-block multiply
- [x] CSR5 format and SpMV (csr5.rs, spmv_csr5.rs) -- tile-based CSR variant with Csr5Matrix, two-phase SpMV (tile + calibration)
- [x] Graph coloring for parallel ILU/IC (graph_coloring.rs) -- distance-2 greedy coloring, parallel_ilu0
- [x] Multi-level ILU(k) (preconditioner/iluk.rs) -- symbolic + numeric with configurable fill levels
- [x] Sparse matrix reordering (reorder.rs) -- RCM and AMD ordering
- [x] Merge-based SpGEMM (ops/spgemm_merge.rs) -- load-balanced with merge-path
- [x] SpMV auto-format selection (ops/auto_spmv.rs) -- heuristic CSR/ELL/BSR/CSR5 selection
- [x] SpGEMM memory estimation (ops/spgemm_estimate.rs) -- upper bound, exact, sampling strategies

### oxicuda-solver (40 files, 13,981 SLoC)
- [x] Solver handle (handle.rs)
- [x] Error types (error.rs)
- [x] PTX helpers (ptx_helpers.rs)
- [x] Dense solvers
  - [x] LU factorization (dense/lu.rs)
  - [x] QR factorization (dense/qr.rs)
  - [x] SVD -- Singular Value Decomposition (dense/svd.rs)
  - [x] Cholesky factorization (dense/cholesky.rs)
  - [x] Eigenvalue decomposition (dense/eig.rs)
  - [x] Matrix inverse (dense/inverse.rs)
  - [x] Determinant computation (dense/det.rs)
  - [x] Least squares (dense/lstsq.rs)
  - [x] Matrix functions (dense/matrix_functions.rs) -- expm, logm, sqrtm via Pade approximation
- [x] Sparse / iterative solvers
  - [x] Conjugate Gradient (sparse/cg.rs)
  - [x] BiCGSTAB (sparse/bicgstab.rs)
  - [x] GMRES (sparse/gmres.rs)
  - [x] Direct sparse solver (sparse/direct.rs)
- [x] Helper utilities
  - [x] Condition number estimation (helpers/condition.rs)
  - [x] Pivoting strategies (helpers/pivot.rs)
- [x] Batched LU/QR/Cholesky (batched.rs) -- BatchedSolver for many small matrices (4x4 to 64x64), batched_solve
- [x] Randomized SVD (randomized_svd.rs) -- Halko-Martinsson-Tropp 2011, configurable rank/oversampling/power iterations
- [x] Preconditioned CG/GMRES (preconditioned.rs) -- Preconditioner trait, Jacobi, PCG, PGMRES
- [x] Tridiagonal/pentadiagonal solvers (tridiagonal.rs) -- Thomas algorithm, cyclic reduction, batched
- [x] Divide-and-conquer SVD (dense/dc_svd.rs) -- recursive bidiagonal splitting
- [x] Symmetric indefinite factorization (dense/ldlt.rs) -- Bunch-Kaufman LDL^T
- [x] Flexible GMRES (sparse/fgmres.rs) -- variable preconditioner per iteration
- [x] Band matrix solvers (dense/band.rs) -- banded LU, Cholesky, solve

### oxicuda-rand (27 files, 9,064 SLoC)
- [x] RNG generator handle (generator.rs) -- unified RNG interface
- [x] Error types (error.rs)
- [x] RNG engines
  - [x] Philox 4x32-10 counter-based RNG (engines/philox.rs)
  - [x] MRG32k3a combined multiple recursive (engines/mrg32k3a.rs) -- with matrix power skip-ahead for parallel MC
  - [x] XORWOW (engines/xorwow.rs)
- [x] Distributions
  - [x] Uniform (distributions/uniform.rs)
  - [x] Normal / Gaussian (distributions/normal.rs)
  - [x] Log-Normal (distributions/log_normal.rs)
  - [x] Poisson (distributions/poisson.rs)
- [x] Quasi-random sequences
  - [x] Sobol sequences (quasi/sobol.rs)
- [x] Philox kernel optimization (philox_optimized.rs) -- 4 values per thread, grid-stride loop, Box-Muller pair generation
- [x] Scrambled Sobol sequences (scrambled_sobol.rs) -- Owen's scrambling for improved equidistribution
- [x] Binomial distribution (distributions/binomial.rs) -- direct inversion + BTPE algorithm
- [x] Geometric distribution (distributions/geometric.rs) -- inverse CDF method
- [x] Halton sequences (quasi/halton.rs) -- multi-dimensional quasi-random
- [x] Latin Hypercube sampling (quasi/latin_hypercube.rs) -- stratified space-filling design
- [x] Multinomial distribution (distributions/multinomial.rs) -- conditional-binomial decomposition
- [x] Truncated normal (distributions/truncated_normal.rs) -- accept-reject Box-Muller

### oxicuda (umbrella crate) (44 files, 18,764 SLoC)
- [x] Re-exports all sub-crates under unified namespace
- [x] ComputeBackend trait (backend.rs) -- ComputeBackend trait, CudaBackend implementation, feature-gated for SciRS2 integration; alloc/free/copy_htod/copy_dtoh/synchronize/init now use real oxicuda_driver calls (PrimaryContext + cuMemAlloc_v2/cuMemcpyHtoD_v2/cuMemcpyDtoH_v2/cuCtxSynchronize); reduce/unary/binary pending PTX pipeline wiring
  - [x] Global initialization (global_init.rs) -- OxiCudaRuntime singleton with device auto-selection
  - [x] OxiONNX GPU inference backend (onnx_backend/) -- IR graph, op implementations, executor, planner, fusion, shape inference
  - [x] ToRSh GPU backend (tensor_backend/) -- tensor, dtype, autograd, ops, optimizer, mixed precision
  - [x] TrustformeRS Transformer GPU backend (transformer_backend/) -- KV-cache, attention, scheduler, speculative decoding, sampling, quantization

---

## Vol.6: Signal Processing, Audio & Image Primitives [COMPLETE]

### oxicuda-signal (13 files, ~3,500 SLoC, 231 tests)

GPU-accelerated signal processing: DCT, DWT, MDCT, STFT, window functions, FIR/IIR filters, image processing — all with CPU reference implementations and PTX kernel generators.

- [x] **Core scaffolding**
  - [x] Error types (error.rs) -- SignalError with 6 variants, SignalResult alias
  - [x] Handle (handle.rs) -- SignalHandle with SmVersion + stream
  - [x] Types (types.rs) -- WaveletFamily, WindowType, SignalPrecision, PadMode
  - [x] PTX helpers (ptx_helpers.rs) -- ptx_header, global_tid_1d, bounds_check, next_pow2
- [x] **DCT transforms** (dct/)
  - [x] DCT-II CPU reference + twiddle PTX kernel (dct2.rs)
  - [x] DCT-III CPU reference + pre-twiddle/un-permute PTX kernels (dct3.rs)
  - [x] DCT-IV CPU reference + PTX pre/post twiddle (dct4.rs)
  - [x] MDCT / IMDCT + sine window + KBD window + MdctPlan (mdct.rs)
- [x] **DWT wavelets** (dwt/)
  - [x] Haar forward/inverse CPU + PTX kernels (haar.rs)
  - [x] Daubechies db2–db10 forward/inverse CPU (daubechies.rs) -- filter tables, conv_downsample
  - [x] Symlet sym2–sym10 forward (sym.rs)
  - [x] Multi-level DWT: forward/inverse + WaveletDecomposition, soft/hard/universal threshold (multilevel.rs)
- [x] **Audio processing** (audio/)
  - [x] STFT / windowed DFT + StftConfig + magnitude/power spectrogram (stft.rs)
  - [x] Window functions: Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Bartlett, Gaussian, FlatTop, Dolph-Chebyshev (stft.rs)
  - [x] Mel filterbank + MFCC + delta/delta-delta coefficients (mel.rs)
  - [x] Audio spectrogram metrics: SNR, peak, LUFS, spectral centroid/rolloff/flatness, MFCC distance (spectrogram.rs)
- [x] **Window analysis** (window.rs)
  - [x] Coherent gain, ENBW, process gain, peak sidelobe level
  - [x] PTX window-apply kernel (element-wise multiply)
  - [x] Standard window catalog
- [x] **FIR/IIR filters** (filter/)
  - [x] FIR design: lowpass, highpass, bandpass, bandstop (windowed sinc) + raised cosine / RRC (fir.rs)
  - [x] FIR application: direct-form with zero/circular/reflect/replicate padding + freq response
  - [x] FIR PTX kernel (direct-form short filter ≤ 64 taps)
  - [x] IIR Biquad sections: lowpass, highpass, bandpass, peaking EQ + freq response (iir.rs)
  - [x] General-order IIR apply (Direct Form II Transposed)
  - [x] Butterworth pole design + SOS cascade
  - [x] Wiener filter: spectral estimation + gain computation + batch apply (wiener.rs)
- [x] **Correlation** (correlation/)
  - [x] Cross-correlation, autocorrelation, normalized correlation coefficient (crosscorr.rs)
  - [x] Phase correlation (sub-pixel peak estimation) for image alignment
  - [x] GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- [x] **Image processing** (image/)
  - [x] Separable Gaussian blur: 1D kernel generation + H/V pass + full 2D blur + PTX kernels (gaussian_blur.rs)
  - [x] Sobel edge detection: Gx/Gy/magnitude/angle + PTX kernels (sobel.rs)
  - [x] Morphological operations: dilate/erode/open/close/top-hat/gradient + structuring elements (morphology.rs)
  - [x] Non-Maximum Suppression: bounding box IoU + greedy NMS + soft-NMS + heatmap NMS (nms.rs)

---

## Vol.7: Computation Graph Engine [COMPLETE]

### oxicuda-graph (11 files, ~4,800 SLoC, 175 tests)

High-level DAG-based computation graph engine that sits above the raw CUDA driver.
Models GPU workloads as computation graphs, applies analysis and optimisation passes,
and lowers to `oxicuda_driver::graph::Graph` for low-overhead CUDA graph submission.

- [x] **Core scaffolding**
  - [x] Error types (error.rs) -- GraphError (10 variants), GraphResult alias
  - [x] Node types (node.rs) -- NodeId, BufferId, StreamId, KernelConfig, MemcpyDir, NodeKind (8 variants), GraphNode, BufferDescriptor
  - [x] ComputeGraph DAG (graph.rs) -- adjacency-list DAG, eager cycle detection, topo sort (Kahn), reachability, DOT export
  - [x] GraphBuilder API (builder.rs) -- fluent builder with auto data-flow edge inference from buffer I/O annotations
- [x] **Analysis passes** (analysis/)
  - [x] Topological analysis (topo.rs) -- ASAP/ALAP scheduling, slack, critical path, level assignment, priority ordering
  - [x] Liveness analysis (liveness.rs) -- buffer live intervals [def_pos, last_use_pos] in topo order, interference pairs, peak live bytes
  - [x] Dominance analysis (dominance.rs) -- Cooper et al. dominator tree, idom, dominates(), dominated_by(), LCA
- [x] **Optimisation passes** (optimizer/)
  - [x] Operator fusion (fusion.rs) -- greedy chain fusion of compatible element-wise kernels using dominator + config checks
  - [x] Memory planning (memory.rs) -- live-interval graph colouring (best-fit), 256-byte alignment, pool layout
  - [x] Stream partitioning (stream.rs) -- list-scheduling heuristic, predecessor-aware stream assignment, cross-stream sync detection
- [x] **Executor backends** (executor/)
  - [x] ExecutionPlan (plan.rs) -- full compilation pipeline (topo→liveness→fusion→memory→stream→linearise), PlanStep sequence with event record/wait pairs
  - [x] Sequential executor (sequential.rs) -- CPU-side simulation, event validity checking, ExecutionStats
  - [x] CUDA graph capture (cuda_graph.rs) -- converts ExecutionPlan to oxicuda_driver::graph::Graph with dependency edges

---

## Vol.8: GPU Training Engine [COMPLETE]

### oxicuda-train (15 files, ~4,200 SLoC, 105 tests)

Production-grade GPU-accelerated training utilities implementing the v1.2 roadmap items:
gradient checkpointing, mixed-precision optimizer states, and large-scale distributed training.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `TrainError` (12 variants) with `TrainResult<T>` alias
  - [x] `TrainHandle` — wraps `Arc<Context>` + `Arc<Stream>` + SM version metadata; `device_sm_version()` via CUdevice_attribute
- [x] **PTX update kernels** (ptx_kernels.rs)
  - [x] `adam_update_ptx` — fused moment update + bias-corrected Adam step; `fma.rn.f32`, `sqrt.approx.f32`, `rcp.approx.f32`
  - [x] `adamw_update_ptx` — decoupled weight decay: `p *= (1 − lr·wd)` before moment update
  - [x] `sgd_update_ptx` — Nesterov SGD with `setp.ne.f32` predicate for conditionality
  - [x] `lion_update_ptx` — sign via bit-mask: `and.b32 sign_bit, c_bits, 0x80000000`
  - [x] `came_row_factor_ptx` / `came_col_factor_ptx` — per-row/col accumulation for CAME factored second moment
  - [x] `norm_sq_partial_ptx` — block-level ‖g‖² with warp butterfly `shfl.sync.bfly.b32` + smem merge
  - [x] `scale_inplace_ptx` / `add_inplace_ptx` — element-wise scale and gradient accumulation
  - [x] All kernels: grid-stride `$LOOP`/`$DONE`, sm_80/sm_90 PTX header selection, `f32_hex()` IEEE literals
- [x] **GPU optimizers** (gpu_optimizer/)
  - [x] `GpuAdam` — bias-corrected first+second moments, optional AMSGrad variant
  - [x] `GpuAdamW` — decoupled weight decay (default wd=0.01); differs from Adam's L2 regularization
  - [x] `GpuLion` — single moment buffer; sign update `p = p·(1−lr·λ) − lr·sign(c)`; 50% memory vs Adam
  - [x] `GpuCame` — factored second moment: `CameV::Matrix { row, col }` O(m+n) vs O(mn) for Adam
  - [x] `GpuMuon` — Nesterov + Newton-Schulz orthogonalisation; 5-iteration `X ← 1.5X − 0.5X·XᵀX`
  - [x] `GpuOptimizer` trait: `step()`, `zero_grad()`, `lr()`, `set_lr()`, `name()`
  - [x] `adam_bias_corrections()` helper: pre-computes `step_size = lr/(1−β₁ᵗ)` and `bc2_rsqrt = 1/√(1−β₂ᵗ)`
- [x] **Gradient utilities** (grad_clip.rs, grad_accum.rs)
  - [x] `GlobalNormClip` — joint ‖g‖ across all params; f64 accumulation; scale = max_norm/(norm+ε)
  - [x] `PerLayerClip` / `ValueClip` — independent per-param and element-wise clipping
  - [x] `GradientAccumulator` — k micro-batch accumulation; `Average` and `Sum` reduction modes
- [x] **Gradient checkpointing** (checkpoint.rs)
  - [x] `CheckpointPolicy`: Uniform { interval }, Selective { names }, Offload, None
  - [x] `CheckpointManager` — save/retrieve/recompute activation segments; `RecomputeFn` closures
  - [x] `CheckpointOverflow` error when segment count exceeds `max_segments`
- [x] **LR Schedulers** (lr_scheduler.rs) — 11 variants via `LrScheduler` trait
  - [x] ConstantLR, StepLR, MultiStepLR, ExponentialLR (with `base_lr()` getters)
  - [x] CosineAnnealingLR, LinearWarmup, WarmupCosine, PolynomialDecayLR, OneCycleLR, CyclicLR
  - [x] ReduceLROnPlateau — metric-based reduction with patience and min_lr floor
- [x] **ZeRO distributed optimizer** (zero.rs)
  - [x] `ZeroStage`: Stage1/2/3 with `shard_range(n) = (rank*chunk, min(start+chunk, n))`
  - [x] `ZeroOptimizer<O: GpuOptimizer>` — wraps any optimizer; Stage2 zeros non-owned gradients; Stage3 operates only on owned parameter shard
  - [x] `ZeroMemoryEstimate` — `bytes_per_rank()` and `reduction_ratio()` capacity planning helpers
- [x] **Integration tests** (lib.rs) — 6 E2E tests: AdamW+WarmupCosine+clip, Lion+accumulation, CAME+CyclicLR, Muon+ReduceLROnPlateau, ZeRO-2, checkpoint+recompute

---

## Vol.9: GPU-Accelerated Reinforcement Learning [COMPLETE]

### oxicuda-rl (29 files, ~6,090 SLoC, 165 tests)

First-class GPU-ready RL library implementing every major modern algorithm from DQN to SAC/TD3/PPO.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `RlError` (12 variants): DimensionMismatch, InsufficientTransitions, InvalidPriority, InvalidConfig, EmptyBatch, NanEncountered, InvalidLogProb, NanLoss, EpisodeError, InvalidStateSize, InvalidAction, Other
  - [x] `SmVersion(u32)` with `ptx_version_str()` mapping sm≥100→"8.7", sm≥90→"8.4", sm≥80→"8.0", else "7.5"
  - [x] `LcgRng` — 64-bit LCG (multiplier 6364136223846793005) with `next_u32()`, `next_f32()`, `next_usize(n)`
  - [x] `RlHandle::default_handle()` — sm=80, device=0, seed=42
- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU kernels
  - [x] `td_error_ptx` — TD-error `δ = r + γ*(1-done)*V' - V`, grid-stride
  - [x] `normalize_advantages_ptx` — mean/variance normalisation pass
  - [x] `ppo_ratio_ptx` — clipped importance ratio `exp(lp_new - lp_old)` with `ex2.approx.f32`
  - [x] `sac_target_ptx` — soft Bellman target `y = r + γ*(1-done)*(min(Q1,Q2) - α*lp)`
  - [x] `per_is_weight_ptx` — IS weight `(N*p_i)^{-β}` normalised by max; `lg2.approx.f32`
- [x] **Experience replay buffers** (buffer/)
  - [x] `UniformReplayBuffer` — struct-of-arrays circular buffer; rejection sampling without replacement
  - [x] `PrioritizedReplayBuffer` — dual sum+min segment tree O(log N); stratified sampling across strata; IS weight computation
  - [x] `NStepBuffer` — circular buffer of `Option<Step>`; n-step return accumulation with γ^n bootstrap; flush on episode end
- [x] **Policy distributions** (policy/)
  - [x] `CategoricalPolicy` — Gumbel-max sampling; log-prob; entropy; KL-divergence; greedy; log_prob_batch
  - [x] `GaussianPolicy` — Box-Muller N(0,1); reparameterisation μ+σ⊙ε; Tanh squashing with Jacobian correction; log-prob batch
  - [x] `DeterministicPolicy` — DDPG exploration noise; TD3 target policy smoothing (clipped Gaussian); `OrnsteinUhlenbeck` OU process
- [x] **Return / advantage estimators** (estimator/)
  - [x] `compute_gae` — backward scan `A_t = δ_t + γλ(1-done)A_{t+1}`; optional Welford normalisation; GAE-λ
  - [x] `compute_td_lambda` — `G_t = r_t + γ*mask*[(1-λ)*v_{t+1} + λ*G_{t+1}]`; takes values[T+1] bootstrap
  - [x] `compute_vtrace` — IMPALA V-trace: c_t=min(c̄,ρ_t), ρ̄_t=min(ρ̄,ρ_t); backward scan advantages
  - [x] `compute_retrace` — safe off-policy Q-targets: c_t=λ*min(1,ρ_t); `Q^ret_t = Q_t + δ_t + γ*c_{t+1}*(Q^ret_{t+1}-Q_{t+1})`
- [x] **RL algorithm loss functions** (loss/)
  - [x] `ppo_loss` — clip ratio*A + `PpoConfig{clip_eps=0.2, c_v=0.5, c_e=0.01}`; approx_kl, clip_fraction metrics
  - [x] `dqn_loss` / `double_dqn_loss` — Bellman MSE/Huber (kappa=1.0); IS-weighted; Double-DQN decoupled selection
  - [x] `sac_critic_loss` / `sac_actor_loss` / `sac_temperature_loss` — entropy-regularized; log-space temperature
  - [x] `td3_critic_loss` / `td3_actor_loss` — twin-Q Bellman error + deterministc actor `-mean(Q1_μ)`
- [x] **Normalization** (normalize/)
  - [x] `RunningStats` — Welford online N-dim: δ=x-mean; mean+=δ/n; M2+=δ*δ₂; dim() accessor; batch update
  - [x] `ObservationNormalizer` — wraps RunningStats; clip=5.0; enable/disable; eval mode (no stat update)
  - [x] `RewardNormalizer` — `ReturnNorm` (G_t=γG_{t-1}+r_t, divide by std), `Clip`, `None` modes; n_envs parallel
- [x] **Environment abstractions** (env/)
  - [x] `Env` trait — `obs_dim()`, `act_dim()`, `reset()`, `step()`, `is_continuous()`
  - [x] `LinearQuadraticEnv` — s'=0.9s+a+noise, r=-s²-0.1a²; Box-Muller noise (>> 41 bit shift, NaN-safe)
  - [x] `VecEnv<E: Env>` — batched `reset_all()`, `step()` (auto-reset on done), `foreach()`, `terminal_obs` tracking
- [x] **Integration tests** (lib.rs) — 5 E2E tests
  - [x] `e2e_dqn_style_loop` — collect 200 transitions + DQN loss on LQ env
  - [x] `e2e_ppo_gae_loss` — 128-step GAE + PPO clip+value+entropy loss
  - [x] `e2e_sac_style_update` — PER buffer + SAC critic loss with IS weights
  - [x] `e2e_vecenv_with_obs_norm` — 4×VecEnv 20 steps + ObservationNormalizer Welford update
  - [x] `e2e_n_step_buffer` — 3-step return verification: R ≈ 1+0.99+0.99²

---

## Vol.10: Quantization & Model Compression Engine [COMPLETE]

### oxicuda-quant (24 files, ~5,442 SLoC, 151 tests)

Post-training quantization (PTQ), quantization-aware training (QAT), pruning,
knowledge distillation, and mixed-precision analysis for LLM and DNN deployment.

- [x] **Error types** (error.rs) — 12 `QuantError` variants
  - DimensionMismatch, EmptyInput, InvalidScale, InvalidBitWidth, GroupSizeMismatch, CalibrationRequired, SingularHessian, TeacherStudentMismatch, AllZeroPruning, NonFiniteFp8, InfeasibleCompressionTarget, InvalidConfig
- [x] **PTX kernels** (ptx_kernels.rs) — 5 GPU-side quantization kernels
  - `fake_quant_ptx` — STE-aware fake quantization
  - `int8_quant_ptx` / `int8_dequant_ptx` — INT8 quant/dequant with scale+zp
  - `nf4_dequant_ptx` — NF4 lookup table in shared memory
  - `prune_mask_ptx` — apply sparsity mask in-place
- [x] **Quantization schemes** (scheme/)
  - [x] `MinMaxQuantizer` (minmax.rs) — INT4/INT8 Symmetric/Asymmetric PerTensor/PerChannel/PerGroup; 9 tests
  - [x] `Nf4Quantizer` (nf4.rs) — QLoRA NF4 with exact quantile LUT, nibble packing, absmax blocks; 8 tests
  - [x] `Fp8Codec` (fp8.rs) — E4M3 (max=448) and E5M2 (max=57344) via IEEE 754 bit manipulation; 8 tests
  - [x] `GptqQuantizer` (gptq.rs) — Hessian OBC via Cholesky + L⁻¹, column-wise weight correction; 8 tests
  - [x] `SmoothQuantMigrator` (smooth_quant.rs) — α-scaled activation/weight migration, preserves output; 8 tests
- [x] **QAT observers and fake quant** (qat/)
  - [x] `MinMaxObserver` — running global min/max, compute scale/zp; 5 tests
  - [x] `MovingAvgObserver` — EMA momentum update of min/max; 3 tests
  - [x] `HistogramObserver` — histogram + min-MSE percentile clipping search; 3 tests
  - [x] `FakeQuantize` — quantize→dequantize forward, STE backward; enabled/disabled mode; 10 tests
- [x] **Pruning** (pruning/)
  - [x] `SparseMask` — boolean weight mask; sparsity(), apply(), apply_in_place(), and/or compose; 7 tests
  - [x] `MagnitudePruner` — L1/L2 unstructured with grouped variant; 8 tests
  - [x] `StructuredPruner` — channel/filter/head granularity, L2-norm unit ranking; 6 tests
- [x] **Knowledge distillation** (distill/)
  - [x] `DistilLoss` — KL(τ²-scaled), MSE, cosine, combined; 10 tests
  - [x] `ResponseDistiller` — soft+hard label combination, batch loss; 5 tests
  - [x] `FeatureDistiller` — per-layer weighted feature matching, normalise_weights; 7 tests
- [x] **Compression analysis** (analysis/)
  - [x] `SensitivityAnalyzer` — per-layer MSE across bit-widths via MinMax symmetric; 8 tests
  - [x] `CompressionMetrics` + `ModelCompressionMetrics` — bits, ratio, sparsity, weighted MSE; 7 tests
  - [x] `MixedPrecisionPolicy` — greedy sensitivity-guided bit assignment; 7 tests

---

## Vol.11: High-Performance Inference Engine [COMPLETE]

### oxicuda-infer (22 files, ~5,900 SLoC, 138 tests)

vLLM-style continuous batching inference engine with PagedAttention KV cache,
speculative decoding, beam search, and a pluggable ModelRunner abstraction.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `InferError` (15 variants): BlockAllocFailed, InvalidSequenceId, EmptyBatch, DimensionMismatch, SamplingError, SchedulerFull, NoPrefillSeqs, CacheManagerError, InvalidSamplingParams, EosTokenMissing, BeamSearchError, SpeculativeError, UnsupportedConfig, ModelRunnerError, Other
  - [x] `InferHandle` — device, sm_version, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, block_size, max_seq_len; `ptx_version_str()`, `attention_scale()`

- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU-side inference kernels
  - [x] `paged_attn_ptx` — online Flash-Attention-style softmax over paged KV blocks; per-block numerically-stable `m_new = max(m, tile_max)`
  - [x] `rope_apply_ptx` — in-place RoPE with `cos.approx.f32` / `sin.approx.f32`; frequency `θ_i = position * 10000^{-2i/d}`
  - [x] `top_k_filter_ptx` — sets non-top-K logit positions to NEG_INFINITY; register-shuffle warp sort
  - [x] `logits_softmax_ptx` — three-pass stable softmax: max→sum_exp→normalize using warp butterfly reduces
  - [x] `kv_append_ptx` — writes K/V into physical block slot; grid-stride across attention heads

- [x] **KV cache** (cache/)
  - [x] `BlockId(u32)` opaque identifier; `KvBlock` with `append()`, `key_slice()`, `value_slice()`, `reset()`
  - [x] `PagedKvCache` — `[n_layers][n_blocks]` 2D block pool; O(1) free-list alloc; reference counting for copy-on-write prefix sharing; 8 tests
  - [x] `CacheManager` — per-sequence block tables `HashMap<u64, Vec<BlockId>>`; auto-grow on block fill; `allocate_sequence`, `free_sequence`, `append_token`; 7 tests
  - [x] `PrefixCache` — FNV-1a token hash → `PrefixEntry`; LRU eviction; `lookup()`, `insert()`, `hit_rate()`; 9 tests

- [x] **Batch scheduling** (batch/)
  - [x] `SequenceStatus`: Waiting → Prefill → Decode → Finished(FinishReason) with EosToken(u32) / MaxLength variants
  - [x] `SamplingParams` — temperature, top_k, top_p, max_new_tokens, eos_token_id, repetition_penalty; 8 tests
  - [x] `Scheduler` — FCFS admission; token-budget decode phase; memory-pressure preemption; `ScheduledBatch{prefill_ids, decode_ids}`; `on_step_complete` / `take_finished`; 9 tests
  - [x] `ContinuousBatcher` — orchestrates scheduler + cache_manager + model_fn + Rng; one batched forward pass per `step()`; 8 tests

- [x] **Sampling suite** (sampling/)
  - [x] `Rng` — 64-bit LCG (Knuth constants); `next_u64()`, `next_f32()`, `next_usize(n)`; `softmax` + `categorical_sample`; 5 tests
  - [x] `greedy_sample` / `greedy_sample_batch` — argmax with NaN guard; 7 tests
  - [x] `top_k_filter` / `top_k_sample` — threshold from k-th sorted logit; exactly-k tokens retained; 7 tests
  - [x] `top_p_filter` / `top_p_sample` — sorted cumulative-probability nucleus cutoff; 5 tests
  - [x] `BeamSearchState::step()` — log-softmax expansion; keep beam_width candidates; EOS → completed; length-normalised `score/len^α`; 8 tests
  - [x] `speculative_verify()` — rejection sampling: accept `d_i` if `u < min(1, p_target/p_draft)`; correction token from `max(0,p−q)/Z`; provably identical distribution to target; 6 tests

- [x] **Executor** (executor/)
  - [x] `ModelRunner` trait — `vocab_size()`, `decode(token_ids, block_tables, seq_lens)`, `prefill(token_ids, seq_starts, block_tables)`; 9 tests
  - [x] `MockModelRunner` — peaks at `(token_id + bias) % vocab_size`; deterministic for unit testing
  - [x] `RunnerStats` — n_steps, total_tokens, sequences_completed; `avg_batch_size()`
  - [x] `paged_attention_cpu` — reference GQA PagedAttention: load K/V per block, Q·K^T·scale, stable softmax, weighted ×V; kv_h=h/(n_heads/n_kv_heads); `AttentionConfig`; 5 tests

- [x] **Integration tests** (lib.rs) — 6 E2E tests
  - [x] `e2e_greedy_until_eos` — continuous batching generates until EOS token
  - [x] `e2e_max_tokens_termination` — max_new_tokens=1 path
  - [x] `e2e_beam_search_completes` — beam_width=2 finishes on EOS in one step
  - [x] `e2e_speculative_all_accepted` — draft==target → all k drafts accepted
  - [x] `e2e_paged_attention_single_token` — Q=V=1.0 → output=1.0
  - [x] `e2e_prefix_cache_hit_rate` — 3 queries (1 miss, 2 hits) → hit_rate=2/3

---

## Vol.12: Distributed Inference Engine [COMPLETE]

### oxicuda-dist-infer (20 files, ~5,800 SLoC, 80 tests)

Multi-GPU distributed inference with three orthogonal parallelism axes (TP × SP × EP = world_size),
distributed KV-cache management, and affinity-aware request routing.

- [x] **Error types and session handle** (error.rs, handle.rs)
  - [x] `DistInferError` (27 variants): InvalidWorldSize, RankOutOfRange, TooFewRanks, TpFeaturesMisaligned, TpInputMisaligned, ShardShapeMismatch, SpSeqLenMisaligned, EmptyChunk, EpExpertsMisaligned, EmptyExpertBatch, SequenceNotOwned, MigrationTargetInvalid, BlockPoolExhausted, AllRanksAtCapacity, EmptyTokenSequence, NoPrefixAffinity, DimensionMismatch, Internal
  - [x] `ParallelismConfig { tp, sp, ep }` — three-way parallelism decomposition, `world_size()`, `validate()`
  - [x] `RankCoordinates` — 3-D tp/sp/ep coordinates from flat global rank; `peer_tp/sp/ep()` for ring lookups
  - [x] `DistInferHandle` — lightweight descriptor with device, SM version, config, coords; `single_rank()` for tests

- [x] **PTX kernel sources** (ptx_kernels.rs) — 5 GPU-side kernels
  - [x] `tp_col_scatter_ptx` — column-parallel linear scatter: write strided shard into full output buffer
  - [x] `tp_row_all_reduce_ptx` — row-parallel linear all-reduce: ring partial-sum accumulation
  - [x] `sp_seq_chunk_copy_ptx` — sequence chunk copy: extract/insert contiguous token slice (direction=0/1)
  - [x] `ep_token_scatter_ptx` — expert-parallel token scatter: route tokens to expert-local input buffers
  - [x] `ep_token_gather_ptx` — expert-parallel token gather: collect expert outputs back to original order

- [x] **Tensor parallelism** (tensor_parallel/)
  - [x] `ColumnLinearShard` — weight shard `[local_out × in]`; `forward()` local GEMM; `validate()`
  - [x] `ColumnLinear` — column-parallel linear: `from_full_weight()` slices rows; `local_forward()`; `all_gather()` simulates collective
  - [x] `RowLinearShard` — weight shard `[out × local_in]`; `forward_partial()` local GEMM; bias only on rank 0
  - [x] `RowLinear` — row-parallel linear: `from_full_weight()` slices columns; `slice_input()`; `all_reduce()` simulates ring reduce

- [x] **Sequence parallelism** (sequence_parallel/)
  - [x] `ChunkInfo` — describes rank's token window: start, len, total_tokens, hidden_dim
  - [x] `SeqSplitter` — `extract_chunk()`, `insert_chunk()`, `all_gather()`, `reduce_scatter()`; validates divisibility
  - [x] `BoundaryExchange` — pre-attention all-gather of K/V; post-attention reduce-scatter of outputs; `local_attention()` with causal masking and GQA-compatible head indexing

- [x] **Expert parallelism** (expert_parallel/)
  - [x] `TopKRouter` — top-K selection from gating logits + softmax weight normalisation; `RoutingPlan` with expert_load; `load_balance_cv()` metric
  - [x] `RoutingEntry` / `RoutingPlan` — per-(token,expert) assignment with routing weight
  - [x] `LocalExpertBatch` — dispatched token batch per expert with token_indices and weights
  - [x] `ExpertDispatcher` — `scatter()` → local expert buffers; `gather()` → weighted output sum; `dispatch_and_gather()` end-to-end

- [x] **Distributed KV cache** (distributed_cache/)
  - [x] `SeqOwnership` / `RankCacheStats` — per-sequence owner rank + block count; per-rank utilization stats
  - [x] `CachePartition` — least-loaded assignment; `grow()`, `release()`; `rebalance_suggestions()` (utilization-threshold migration hints); `apply_migration()`
  - [x] `BlockData` — serialized KV block `[n_layers × 2 × block_size × kv_dim]`; `key_slice(l)` / `value_slice(l)`; `validate()`
  - [x] `MigrationRequest` / `MigrationStats` — cross-rank block transfer descriptor + statistics
  - [x] `BlockMigrator` — `receive_block()` → local staging id; `take_block()`; `validate_target()`; stats tracking

- [x] **Request routing** (router/)
  - [x] `Request` — token_ids, max_new_tokens, priority; `prefix_hash(len)` FNV-1a for affinity lookup
  - [x] `RoutingDecision` / `DispatchPolicy` — selected rank + policy tag + prefix_hit flag
  - [x] `RankLoad` — free_blocks, total_blocks, in_flight; `utilization()`
  - [x] `RouterMetrics` — per-policy request counts, total_routed, prefix_hits, `prefix_hit_rate()`
  - [x] `RoutingPolicy` — three modes: RoundRobin, LeastLoaded, PrefixAffinity (with fallback + registration)

- [x] **Integration tests** (lib.rs) — 6 E2E tests
  - [x] `e2e_tp_column_row_roundtrip` — tp=4 column-parallel + all-gather + row-parallel + all-reduce = identity
  - [x] `e2e_sp_attention_pipeline` — sp=2 extract chunks + all-gather + local_attention (uniform QKV → output=1.0)
  - [x] `e2e_ep_moe_dispatch_gather` — ep=2, 4 experts, 4 tokens, top-1 routing + identity experts + gather
  - [x] `e2e_cache_partition_lifecycle` — 4 ranks, 8 sequences, assign/grow/release lifecycle
  - [x] `e2e_routing_prefix_affinity_pipeline` — first request misses, second with same prefix hits same rank
  - [x] `e2e_ptx_kernels_all_sm_versions` — all 5 kernels × 5 SM versions produce valid PTX headers

---

## Vol.13: LLM Inference Primitives [COMPLETE]

### oxicuda-lm (16 files, ~3,200 SLoC, 182 tests)

Model-layer abstractions for LLM inference: BPE tokenizer, transformer layer building blocks
with KV-cache for incremental decode, complete GPT-2 and LLaMA-2/3 model implementations,
and GPU kernel PTX string generators.

- [x] **Error types and config** (error.rs, config.rs)
  - [x] `LmError` (17 variants): DimensionMismatch, InvalidConfig, EmptyInput, OutOfVocab, Utf8Decode, WeightNotFound/ShapeMismatch, LayerIndexOutOfRange, HeadDimMismatch, KvCacheLengthMismatch, SequenceTooLong, InvalidMergePair, VocabSizeMismatch, GqaHeadMismatch, WeightDataLengthMismatch, Internal
  - [x] `GptConfig` — GPT-2 presets: `gpt2_small` (12L/12H/768D), `gpt2_medium` (24L/16H/1024D), `gpt2_large`, `gpt2_xl`, `tiny` (2L/2H/8D)
  - [x] `LlamaConfig` — LLaMA presets: `llama2_7b`, `llama2_13b`, `llama3_8b` (GQA 32H/8KV), `mistral_7b`, `phi2`, `tiny` (2L/4H/2KV)
  - [x] `SmVersion` + `LmHandle` (handle.rs) — SM version with `ptx_version_str()`, `target_str()`

- [x] **Weights** (weights.rs)
  - [x] `WeightTensor { data, shape }` — `zeros()`, `ones()`, `eye()`, `from_data()`, `row_slice()`, `validate_shape()`
  - [x] `ModelWeights` (HashMap-backed) — `get_checked()` with shape validation, `n_params()`, iterators

- [x] **PTX kernel generators** (ptx_kernels.rs) — 5 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `embedding_forward_ptx` — token embedding table lookup (grid-stride over n_tokens×embed_dim)
  - [x] `rope_apply_ptx` — RoPE in-place from pre-computed cos/sin tables; grid-stride pair indexing
  - [x] `silu_gate_ptx` — SwiGLU gate: `out = (g/(1+exp(-g))) * up`; `ex2.approx.f32` + `rcp.approx.f32`
  - [x] `rms_norm_ptx` — shared-memory warp butterfly reduction → normalize + scale; `sqrt.approx.f32`
  - [x] `causal_attn_softmax_ptx` — per-head causal mask + stable softmax (max → exp → sum → normalize)

- [x] **Tokenizer** (tokenizer/)
  - [x] `Vocab` (vocab.rs) — byte↔id bidirectional map; `gpt2_byte_vocab()` (256 byte tokens); `with_extra_tokens()`, `special_id()`
  - [x] `BpeTokenizer` (bpe.rs) — byte-level BPE; `merge_ranks` (priority table) + `pair_to_merged` (result table); `encode()` vocab-lookup init → greedy lowest-rank merge loop; `decode()` byte concat → UTF-8
  - [x] `BpeBuilder` — `add_merge()`, `add_special()`, `build()` convenience builder

- [x] **Layers** (layer/)
  - [x] `RmsNorm` / `LayerNorm` (norm.rs) — per-token normalize with learnable weight (and bias for LayerNorm)
  - [x] `TokenEmbedding`, `LearnedPositionalEmbedding`, `RotaryEmbedding` (embedding.rs) — RoPE with precomputed cos/sin tables, absolute position offset for KV-cache decode
  - [x] `MlpFfn` (ffn.rs) — GPT-2 GELU MLP: `W_proj(GELU(W_fc·x+b))+b_proj`
  - [x] `SwiGluFfn` (ffn.rs) — LLaMA SwiGLU: `W_down(silu(W_gate·x) ⊙ W_up·x)`, no biases
  - [x] `LayerKvCache` / `MultiHeadAttention` (attention.rs) — GQA (`kv_h = q_h / (n_heads/n_kv_heads)`), causal mask at absolute position `past_len + t`, KV append for incremental decode
  - [x] `GptBlock` / `LlamaBlock` / `PastKvCache` (transformer.rs) — pre-LN residual blocks; multi-layer KV cache container

- [x] **Models** (model/)
  - [x] `Gpt2Model` (model/gpt.rs) — token+pos embedding → N×GptBlock → LayerNorm → weight-tied LM head; `next_token()` greedy decode
  - [x] `LlamaModel` (model/llama.rs) — TokenEmbedding → N×LlamaBlock → RmsNorm → independent LM head; `next_token()` greedy decode
  - [x] Weight loaders (model/weights.rs) — `load_gpt2_block()` (HuggingFace key convention, packed QKV split), `load_llama_block()` (separate q/k/v proj)

- [x] **Integration tests** (lib.rs) — 10 E2E tests
  - [x] GPT-2 tiny forward (shape, zero-weight → zero-logits)
  - [x] LLaMA tiny forward (shape validation)
  - [x] GPT-2 incremental decode consistency (full vs token-by-token last-position logit match)
  - [x] LLaMA incremental decode consistency
  - [x] BPE encode/decode round-trip ("hello" → [259] → "hello")
  - [x] RMSNorm + LayerNorm numerical correctness
  - [x] PTX kernels × 6 SM versions (target directive presence)
  - [x] LLaMA GQA multi-step decode (prefill 4 + decode 3 → past_len=7)
  - [x] Vocab special token round-trip (BOS/EOS)
  - [x] GPT-2 greedy decode loop (5 steps, all IDs in vocab range)

## Vol.14–16: SDE Samplers in oxicuda-rand [COMPLETE]

Added stochastic differential equation (SDE) integration methods as a new `sde` submodule
in `oxicuda-rand`, providing GPU-simulation building blocks for financial models, physical
simulations, and score-based generative model training.

### oxicuda-rand SDE additions (4 files, ~1,060 SLoC, 71 new tests)

- [x] **SDE framework** (sde/mod.rs) — shared `SdeProcess` trait, `SdeConfig`, `PathMatrix`, Xoshiro256** PRNG
- [x] **Brownian motion** (sde/brownian.rs) — `BrownianMotion`, `GeometricBrownianMotion` (exact), `OrnsteinUhlenbeck` (exact), `BrownianPathResult` with covariance check
- [x] **Euler-Maruyama** (sde/euler_maruyama.rs) — strong order 0.5 scheme for `dX = μ dt + σ dW`; `EulerMaruyamaResult` with mean/std/path statistics; `strong_error()` comparison
- [x] **Milstein** (sde/milstein.rs) — strong order 1.0 via `½σσ'(ΔW² − Δt)` correction; `convergence_comparison()` verifying EM vs Milstein accuracy on GBM
- [x] **Stratonovich-Heun** (sde/heun.rs) — predictor-corrector for Stratonovich SDEs; `solve_ito()` with automatic Itô→Stratonovich correction `μ_strat = μ − ½σσ'`

---

## Vol.17: Generative AI Primitives [COMPLETE]

### oxicuda-gen (25 files, ~7,400 SLoC, 221 tests)

Pure-Rust generative AI primitives: diffusion schedulers (DDPM/DDIM/DPM-Solver++/Flow Matching),
classifier-free guidance, VQ-VAE codec, LoRA adapters, and score-network building blocks.

- [x] **Error types** (error.rs) — `GenError` (15 variants): DimensionMismatch, InvalidBetaRange, InvalidGuidanceScale, UnsupportedDpmOrder, InvalidTimestep, InvalidCodebookSize, WeightShapeMismatch, InvalidLoraRank, and more
- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (seed-based, Box-Muller normals), `GenHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 6 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `ddpm_step_ptx` — `x_{t-1} = (x_t − β/√(1−ᾱ) · ε̂)/√α + σ·z` using `sqrt.approx`, `rcp.approx`
  - [x] `cfg_combine_ptx` — `out = u + s·(c − u)` classifier-free guidance blend
  - [x] `lora_apply_ptx` — `y = x·W + (α/r)·x·B·A` low-rank update; grid-stride
  - [x] `flow_velocity_ptx` — Euler step `x_{t+Δ} = x_t + Δ·v(x_t, t)` for flow ODE
  - [x] `vae_kl_loss_ptx` — `0.5·Σ(μ² + σ² − 1 − log σ²)` latent KL divergence
  - [x] `timestep_embed_ptx` — sinusoidal embedding via `sin/cos/lg2/ex2`

- [x] **Schedulers** (scheduler/) — 5 files
  - [x] `BetaSchedule` (beta_schedule.rs) — linear, cosine (Nichol & Dhariwal), scaled-cosine, sigmoid β schedules; `alphas_bar`, `sqrt_alphas_bar`, `sqrt_one_minus_alphas_bar`
  - [x] `DdpmScheduler` (ddpm.rs) — `add_noise()` with `q(xₜ|x₀)`, `step()` reverse DDPM update with fixed σ²=βₜ
  - [x] `DdimScheduler` (ddim.rs) — η-parameterised deterministic/stochastic; η=0 → identical two-call results
  - [x] `DpmSolverScheduler` (dpm_solver.rs) — exponential integrator on `λₜ = log(αₜ/σₜ)`; 1st/2nd-order multi-step; `num_train_steps()` accessor
  - [x] `FlowMatchingScheduler` (flow_matching.rs) — linear OT path `xₜ = (1−t)x₀+tx₁`; Euler and Heun ODE solvers; boundary conditions verified in tests

- [x] **Guidance** (guidance/) — 3 files
  - [x] `CfgGuidance` (cfg.rs) — `ε̂ = uncond + s·(cond − uncond)` with scale-clipping and rescaling
  - [x] `PerpNegGuidance` (perp_neg.rs) — perpendicular-negative prompt guidance
  - [x] `AdaptiveCfgScheduler` (adaptive.rs) — constant/linear/cosine/stepwise dynamic scale scheduling

- [x] **VAE** (vae/) — 4 files
  - [x] `GaussianLatent` (kl.rs) — reparameterised sampling `z = μ + ε·σ`, `kl_loss()`, `standard_normal()`
  - [x] `VqCodebook` (quantize.rs) — EMA codebook update `eₖ ← γeₖ + (1−γ)∑xⱼ`, nearest-entry lookup, commitment loss
  - [x] `Encoder` (encoder.rs) — ResNet down-blocks (GELU + GroupNorm); `EncoderWeights::zeros()`
  - [x] `Decoder` (decoder.rs) — mirrored up-sampling blocks; `DecoderWeights::zeros()`

- [x] **LoRA** (lora/) — 2 files
  - [x] `LoraLinear` (adapter.rs) — `W' = W + (α/r)·BA`; B∈ℝᵈˣʳ Gaussian init, A=0 init; `forward()` adds rank-r correction
  - [x] `LoraModel` (adapter.rs) — named adapter collection with `add_adapter()`/`apply()`
  - [x] Weight merging (merge.rs) — `merge_lora()`, `unmerge_lora()`, `verify_merge_roundtrip()`, `scale_adapter()`, `compose_adapters()`

- [x] **Score networks** (score/) — 2 files
  - [x] `SinusoidalEmbedding` / `FourierEmbedding` (timestep.rs) — sin+cos pair embedding with sin²+cos²=1 verified
  - [x] `UNetResBlock`, `SelfAttentionBlock`, `CrossAttentionBlock` (unet_block.rs) — SiLU + time-embedding injection + multi-head attention

- [x] **Integration tests** (lib.rs) — 11 E2E tests covering all modules + PTX × 6 SM versions

---

## Vol.18: Graph Neural Network Primitives [COMPLETE]

### oxicuda-gnn (25 files, ~7,370 SLoC, 233 tests)

Pure-Rust GNN library: sparse graph representations (CSR/COO/heterogeneous), message passing,
GCN/GAT/GATv2/GraphSAGE/GIN layers, global and hierarchical pooling, Set2Set readout.

- [x] **Error types** (error.rs) — `GnnError` (14 variants): EmptyGraph, NodeIndexOutOfRange, EdgeIndexOutOfRange, InvalidLayerConfig, FeatureDimensionMismatch, InvalidEdgeWeight, InvalidPoolingK, SamplingError, and more

- [x] **Handle** (handle.rs) — `SmVersion`, `GnnHandle`, `LcgRng`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions
  - [x] `csr_spmv_ptx` — `y[i] = Σ A[i,j]·x[j]`; warp-per-row with `shfl.sync.down` butterfly reduction
  - [x] `scatter_add_ptx` — `out[idx[i]] += in[i]`; `atom.global.add.f32`
  - [x] `gat_attention_ptx` — `LeakyReLU(aᵀ[Wxᵢ‖Wxⱼ])` per edge
  - [x] `softmax_edge_ptx` — numerically stable per-source softmax over outgoing edges
  - [x] `aggregate_mean_ptx` — accumulator / `degree[i]` mean reduction
  - [x] `gin_combine_ptx` — `(1+ε)·xᵢ + Σxⱼ` self-loop aggregator
  - [x] `topk_score_ptx` — `tanh(pᵀx/‖p‖)` scoring for Top-K node selection

- [x] **Graph representations** (graph/) — 4 files
  - [x] `CsrGraph` (csr.rs) — `row_ptr/col_idx/edge_weight`; `from_edges()`, `neighbors()`, `degrees()`, `normalized_adjacency()` (D̂⁻¹/²ÂD̂⁻¹/²)
  - [x] `CooGraph` (coo.rs) — COO format with `to_csr()` conversion; symmetry detection
  - [x] `HeterogeneousGraph` (heterogeneous.rs) — multi-type node/edge relations
  - [x] `KHopSubgraph`, random walk, Node2Vec biased walk (sampling.rs)

- [x] **Message passing** (message_passing/) — 3 files
  - [x] Aggregations (aggregate.rs) — sum/mean/max/min/softmax over neighbor messages
  - [x] Scatter ops (scatter.rs) — `scatter_add`, `scatter_max`, `scatter_min`, `scatter_mul`, `scatter_softmax`
  - [x] Update functions (update.rs) — MLP (2-layer), identity, ReLU, SiLU, LeakyReLU

- [x] **GNN Layers** (layers/) — 5 files
  - [x] `GcnLayer` (gcn.rs) — `H⁽ˡ⁺¹⁾ = σ(D̂⁻¹/²ÂD̂⁻¹/² H⁽ˡ⁾ W⁽ˡ⁾)` (Kipf & Welling 2017)
  - [x] `GatLayer` (gat.rs) — `αᵢⱼ = softmax(LeakyReLU(aᵀ[Wxᵢ‖Wxⱼ]))`, multi-head concat/mean (Veličković 2018)
  - [x] `GatV2Layer` (gat_v2.rs) — dynamic attention `aᵀLeakyReLU(W[xᵢ‖xⱼ])` (Brody 2022)
  - [x] `SageLayer` (sage.rs) — mean/MaxPool/LSTM aggregators; optional L2-norm output (Hamilton 2017)
  - [x] `GinLayer` (gin.rs) — `(1+ε)·hᵥ + Σhᵤ` with MLP; BatchNorm; trainable ε (Xu 2019)

- [x] **Pooling** (pooling/) — 3 files
  - [x] `GlobalPool` (global_pool.rs) — mean/max/sum/attention pooling to graph-level repr; batched graphs
  - [x] `TopKPool` (topk_pool.rs) — Gao & Ji top-k node selection with `tanh(pᵀx/‖p‖)` scoring
  - [x] `DiffPool` (diff_pool.rs) — differentiable hierarchical: `S=softmax(GNN(A,X))`, `X'=SᵀX, A'=SᵀAS`; LP + entropy regularisation losses

- [x] **Readout** (readout/) — 1 file
  - [x] `Set2Set` (set2set.rs) — LSTM-based permutation-invariant readout: `qₜ=LSTM(q*_{t-1})`, `αᵢₜ=softmax(xᵢᵀqₜ)`, `q*ₜ=[qₜ‖rₜ]` (Vinyals 2016)

- [x] **Integration tests** (lib.rs) — 12 E2E tests covering CSR, COO, scatter, GCN, SAGE, GIN, DiffPool, Top-K, sampling, Set2Set, PTX × 6 SM versions

---

## Vol.19: State Space Model Primitives [COMPLETE]

### oxicuda-mamba (25 files, ~7,800 SLoC, 339 tests)

Pure-Rust SSM library: S4 (HiPPO-LegS / DPLR), Mamba selective scan (S6), Mamba-2 (SSD),
and RWKV time-mixing — linear-time alternatives to attention, zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `MambaError` (15 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidSeqLen, InvalidSsmOrder, InvalidModelDim, NonPositiveDelta, InvalidChunkSize, HeadDimMismatch, WeightShapeMismatch, NonFinite, Internal
- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `MambaHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `selective_scan_ptx` — Mamba S6: `h = Ā·h + B̄·u, y = C·h` per-channel sequential recurrence
  - [x] `parallel_scan_ptx` — Warp-level `(A,b)` associative prefix scan via `shfl.sync.down.b32` butterfly
  - [x] `depthwise_conv1d_ptx` — Causal 1D depthwise conv with zero-pad, `fma.rn.f32`
  - [x] `wkv_forward_ptx` — RWKV WKV with numerically stable running-max pivot; `ex2.approx.f32`
  - [x] `ssd_chunk_ptx` — Mamba-2 SSD chunk: causal `Π A_k` accumulation per output position
  - [x] `hippo_legendre_ptx` — HiPPO-LegS forward Euler `c_n' = c_n·(1−Δ(n+1)) + Δ√(2n+1)·u`
  - [x] `rms_norm_silu_ptx` — Fused RMSNorm + SiLU gate; warp butterfly sum via `shfl.sync.bfly.b32`

- [x] **SSM core** (ssm/) — 3 files
  - [x] `discretize.rs` — ZOH (`Ā = exp(Δ·A)`), Bilinear (Tustin), Euler; L'Hôpital limit for `|A| ≈ 0`
  - [x] `parallel_scan.rs` — `ScanPair {a,b}` with associative `⊕` operator, inclusive/exclusive prefix scan, `ssm_state_scan(a_bar, b_bar_u)`
  - [x] `ssm_kernel.rs` — `SsmKernel`: batch-aware `h[b,t,d,n] = Ā·h_prev + B̄·u` recurrence, ZOH discretization, output `y = Σ C·h`

- [x] **S4 architecture** (s4/) — 3 files
  - [x] `hippo.rs` — `hippo_legs(n)`: HiPPO-LegS A matrix (lower-triangular, `A[n,k]=−√(2n+1)√(2k+1)`) and B vector; `hippo_legs_diag`; `hippo_nplr` NPLR decomposition (`λ[n]=−(n+0.5)`, `p=q=√(n+0.5)`)
  - [x] `dplr.rs` — `Dplr {lambda, p, q}`: `A = diag(λ) − p·qᵀ`; `from_hippo`, `to_dense`, ZOH SSM kernel computation via mode decomposition
  - [x] `s4_layer.rs` — `S4Layer`: multi-channel convolutional mode, `naive_conv1d` O(L²) reference, optional bidirectional averaging, `S4Config` builder

- [x] **Mamba** (mamba/) — 3 files
  - [x] `selective_scan.rs` — `selective_scan`: input-dependent `Δ=softplus(proj)`, `Ā=exp(Δ⊗A)`, `B̄=Δ⊗B_proj`, sequential state recurrence; `softplus` with ±20 stability clamp
  - [x] `mamba_block.rs` — `MambaBlock`: in_proj → x/z split → conv1d+SiLU → selective_scan → D skip → SiLU gate → out_proj + residual; `rms_norm`, `linear`, `silu`, `causal_depthwise_conv1d` helpers
  - [x] `mamba_model.rs` — `MambaModel`: TokenEmbedding → N×MambaBlock → RMSNorm → LM head; `forward` returns logits, `next_token` greedy decode; `MambaConfig::tiny()` test preset

- [x] **Mamba-2 / SSD** (mamba2/) — 3 files
  - [x] `ssd.rs` — `ssd_naive` O(L²·N) semi-separable matrix-vector product; `ssd_recurrent` O(L·N) state form; `verify_ssd_equivalence` agreement check (tol 1e-4)
  - [x] `chunk_scan.rs` — `ChunkConfig` with ceiling-division chunks; `chunk_scan`: intra-chunk naive SSD + inter-chunk boundary state propagation; `verify_chunk_equivalence`
  - [x] `mamba2_block.rs` — `Mamba2Block`: multi-head SSD with `a[t]=sigmoid(−exp(a_h))`, per-head `chunk_scan`, D skip, RMSNorm, out_proj + residual

- [x] **RWKV** (rwkv/) — 3 files
  - [x] `time_mixing.rs` — `WkvState {a,b,p}` recurrent state; numerically stable WKV via running-max pivot; `TimeMixingLayer` full RWKV-4 pipeline: LN → token-shift → r/k/v projection → WKV → sigmoid gate → output projection; `layer_norm`, `sigmoid` helpers
  - [x] `channel_mixing.rs` — `ChannelMixingLayer`: token-shift → sigmoid-gated receptance → Square-ReLU expansion → value contraction; `square_relu(x) = max(0,x)²`
  - [x] `rwkv_block.rs` — `RwkvBlock`: pre-norm residual: `y = x + time_mixing(LN₁(x))`, `y = y + channel_mixing(LN₂(y))`

- [x] **Integration tests** (lib.rs) — 20 E2E tests covering all modules + PTX × 6 SM versions

---

---

## Vol.20: Vision Transformer & CLIP Primitives [COMPLETE]

### oxicuda-vision (25 files, ~7,500 SLoC, 349 tests)

Pure-Rust vision library: ViT patch embedding, multi-head self-attention, CLIP
contrastive learning, image augmentation, FPN multi-scale features, DETR decoder,
and bipartite set matching — zero CUDA SDK dependency.

- [x] **Error types** (error.rs) — `VisionError` (15 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidImageSize, InvalidPatchSize, InvalidEmbedDim, InvalidNumHeads, HeadDimMismatch, InvalidNumClasses, InvalidProjDim, NonPositiveTemperature, InvalidRoiBox, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `VisionHandle`

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `patch_embed_ptx` — Strided Conv2D: `[C, H, W] → [N_patches, embed_dim]` with `fma.rn.f32`
  - [x] `bilinear_interp_ptx` — Sub-pixel 4-tap bilinear sampler with half-pixel convention
  - [x] `contrastive_loss_ptx` — InfoNCE: 3-pass numerically stable row-softmax + diagonal CE
  - [x] `roi_align_ptx` — Per-bin bilinear RoI feature extraction with `sampling_ratio²` sample averaging
  - [x] `image_normalize_ptx` — Channel-wise `(x − mean[c]) / std[c]` in-place
  - [x] `adaptive_avg_pool_ptx` — Adaptive 2D average pool with integer window bounds
  - [x] `focal_loss_ptx` — Focal loss `−α(1−p)^γ log p` via stable sigmoid + log

- [x] **Patch embedding** (patch_embed/) — 2 files
  - [x] `conv2d_patch.rs` — `PatchEmbedConfig`, `PatchEmbedWeights` (Xavier init), `PatchEmbed::forward`, `prepend_cls`
  - [x] `pos_embed.rs` — `pos_2d_sincos` (4-band H/W sinusoidal), `LearnablePosEmbed`, `add_pos_embed`

- [x] **ViT** (vit/) — 3 files
  - [x] `vit_block.rs` — `ViTBlock`: pre-norm MHSA + GELU MLP + residuals; `layer_norm`, `gelu_exact` (tanh approx), `softmax_rows`, `mhsa`
  - [x] `vit_encoder.rs` — `ViTEncoder`: N stacked `ViTBlock` + final LayerNorm
  - [x] `vit_model.rs` — `ViTModel`: PatchEmbed → CLS-prepend → PosEmbed → Encoder → head; `ViTConfig::tiny()` (img=32, p=4, D=64, depth=2, heads=4, classes=10)

- [x] **CLIP** (clip/) — 3 files
  - [x] `vision_encoder.rs` — `ClipVisionEncoder` wrapping `ViTEncoder`, CLS-pool to `[embed_dim]`
  - [x] `projection.rs` — `ProjectionHead`: linear + L2-norm; `cosine_sim`
  - [x] `contrastive.rs` — `info_nce_loss`: symmetric InfoNCE with numerically stable log-sum-exp

- [x] **Augmentation** (augment/) — 3 files
  - [x] `geometric.rs` — `random_crop`, `center_crop`, `random_horizontal_flip`, `resize_bilinear` (half-pixel bilinear)
  - [x] `photometric.rs` — `color_jitter` (brightness/contrast/saturation), `random_grayscale` (YIQ luminance)
  - [x] `normalize.rs` — `normalize_chw`, `IMAGENET_MEAN`/`IMAGENET_STD`; `AugOp` enum + `Pipeline::push` builder

- [x] **FPN** (fpn/) — 2 files
  - [x] `lateral.rs` — `LateralConv1x1`: 1×1 conv channel reduction (Xavier init)
  - [x] `top_down.rs` — `Fpn`: lateral → top-down (nearest upsample + add) → 3×3 smooth conv; `FeatureMap {data, channels, h, w}`

- [x] **Detection** (detection/) — 3 files
  - [x] `roi_align.rs` — CPU reference RoI Align with `bilinear_sample_2d`; validates `x2>x1, y2>y1`
  - [x] `detr_decoder.rs` — `DetrDecoderLayer`: self-attn + cross-attn + FFN (pre-norm); `DetrDecoder` depth stack; `DetrConfig::tiny()`
  - [x] `set_match.rs` — `bipartite_match` (greedy + 2-opt); `build_cost_matrix` (class CE + L1 box + GIoU); `giou`

- [x] **Integration tests** (lib.rs) — 19 E2E tests covering all modules + PTX × 6 SM versions

---

## Vol.21: Audio/Speech ML Architectures [COMPLETE]

### oxicuda-audio (28 files, ~7,500 SLoC, 286 tests)

Pure-Rust audio/speech ML library: Conformer encoder, Wav2Vec2 CNN feature extractor,
CTC forward algorithm + prefix beam search, WaveNet dilated stack, SpecAugment
augmentation, speaker embeddings (x-vector TDNN, attentive pooling) — zero CUDA SDK
dependency.

- [x] **Error types** (error.rs) — `AudioError` (17 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidNumMels, InvalidSequenceLength, InvalidEmbedDim, InvalidNumHeads, HeadDimMismatch, InvalidVocabSize, InvalidBeamWidth, InvalidDilation, InvalidKernelSize, InvalidStride, BlankOutOfRange, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `AudioHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `stride_conv1d_ptx` — Strided 1-D conv for Wav2Vec2 CNN feature extractor
  - [x] `dilated_conv1d_ptx` — Causal dilated conv (WaveNet filter+gate, left-pad)
  - [x] `ctc_alpha_ptx` — Log-domain CTC forward alpha recursion with `log_sum_exp` via `ex2`/`lg2`
  - [x] `spec_augment_mask_ptx` — In-place time+freq masking via `setp`/`selp.f32`
  - [x] `depthwise_conv1d_ptx` — Causal depthwise conv for Conformer conv module
  - [x] `rel_pos_bias_ptx` — Relative-position bias table lookup with `min/max.u32` clamping
  - [x] `stats_pool_ptx` — Two-pass mean+std pooling with warp-shuffle reduction

- [x] **Features** (features/) — 3 files
  - [x] `log_mel_adapter.rs` — `LogMelInput` validated `[T, F]` wrapper for `oxicuda-signal` output
  - [x] `cmvn.rs` — `CmvnConfig`, `compute_cmvn`, `apply_cmvn` (per-channel zero-mean unit-variance)
  - [x] `delta.rs` — `compute_delta`, `compute_delta_delta`, `stack_delta_features` (central-difference, edge-pad)

- [x] **Encoder** (encoder/) — 3 files
  - [x] `wav2vec_cnn.rs` — `Wav2VecCnnEncoder`: 7-layer stride-conv1d + group-norm + GELU; `wav2vec2_base()` and `tiny()` configs
  - [x] `conv_module.rs` — `ConvModule`: LN → PW-expand → GLU → depthwise-causal → BN → Swish → PW-reduce
  - [x] `conformer_block.rs` — `ConformerBlock` (macaron ½·FFN + MHSA(rel-pos) + ConvModule + ½·FFN + LN) + `ConformerEncoder`; `ConformerConfig::tiny()` (D=64, heads=4, depth=2, kernel=15)

- [x] **Attention** (attention/) — 2 files
  - [x] `rel_pos_encoding.rs` — `RelPosEncoding {table: [2*max_len-1]}` with seeded init, `bias(q,k)`, `bias_matrix(Q,K)`
  - [x] `rel_pos_attention.rs` — `RelPosAttention`: multi-head SDPA + relative-position bias pre-softmax

- [x] **CTC** (ctc/) — 2 files
  - [x] `forward.rs` — `ctc_forward_log`: log-domain alpha recursion, extended target `l'=[blank,l0,blank,l1,…]`, `log_sum_exp2` stable
  - [x] `beam_search.rs` — `ctc_beam_search`: CTC prefix beam search with `HashMap<Vec<usize>, (p_blank, p_nb)>` merge and pruning

- [x] **Vocoder** (vocoder/) — 2 files
  - [x] `wavenet_block.rs` — `WaveNetBlock`: dilated-causal-conv → tanh⊙sigmoid gated activation → skip + residual pointwise convs
  - [x] `dilated_stack.rs` — `WaveNetStack`: multi-cycle `[1,2,4,…,512]` dilation schedule + 2-layer ReLU head; `tiny()` and `default_config()`

- [x] **Augmentation** (augment/) — 2 files
  - [x] `spec_augment.rs` — `time_mask`, `freq_mask` (SpecAugment), enum-dispatched `SpecAugOp` + `SpecAugPipeline::push` builder
  - [x] `time_warp.rs` — `time_warp`: single-anchor bilinear time-axis warping (no-op when T ≤ 2·max_w)

- [x] **Speaker** (speaker/) — 3 files
  - [x] `stats_pool.rs` — `stats_pool`: two-pass Bessel-corrected temporal mean+std pooling `[T,C] → [2C]`
  - [x] `attentive_pool.rs` — `AttentivePool`: bottleneck `tanh`-attention softmax over time → weighted mean+std `[2C]`
  - [x] `x_vector.rs` — `XVectorTdnn`: 5-layer dilated TDNN (Snyder 2018), stats pool, 512-d affine; `default_config()` + `tiny()`

- [x] **Integration tests** (lib.rs) — 21 E2E tests covering all modules + PTX × 6 SM versions

---

## Quality Gates

| Metric | Target | Achieved |
|--------|--------|----------|
| Compiler warnings | 0 | 0 |
| Clippy warnings | 0 | 0 |
| unwrap() in library code | 0 | 0 |
| C/Fortran build deps | 0 | 0 |
| Test count | >500 | 8,980+ |
| Test pass rate | 100% | 100% |
| Code lines (SLoC) | >30K | ~302,500 |
| Crate count | 12 | 33 |
| GPU arch coverage | SM 7.5--10.0 | SM 7.5--10.0 |
| Pure Rust | 100% default features | 100% |

---

## Vol.22: Time-Series Forecasting Architectures [COMPLETE]

### oxicuda-timeseries (30 files, ~8,500 SLoC, 177 tests)

Pure-Rust time-series forecasting and classification library: TCN, NHiTS, PatchTST,
TimesNet, iTransformer, RevIN, series decomposition — zero CUDA SDK dependency.
Time-major `[T, C]` layout throughout; all variates channels-last.

- [x] **Error types** (error.rs) — `TsError` (18 variants): DimensionMismatch, ShapeMismatch, EmptyInput, InvalidSequenceLength, InvalidNumVariates, InvalidPatchLen, InvalidStride, InvalidKernelSize, InvalidDilation, InvalidNumHeads, HeadDimMismatch, InvalidEmbedDim, InvalidHorizon, InvalidPoolSize, InvalidTopK, WeightShapeMismatch, NonFinite, Internal

- [x] **Handle** (handle.rs) — `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle), `TsHandle::default_handle()` (SM 8.0, device 0, seed 42)

- [x] **PTX kernels** (ptx_kernels.rs) — 7 GPU kernels × 6 SM versions (75/80/86/90/100/120)
  - [x] `moving_average_ptx` — Strided centred moving average over time axis
  - [x] `patch_embed_1d_ptx` — Extract overlapping 1-D patches [N,T]→[N,num_patches,patch_len]
  - [x] `causal_temporal_conv_ptx` — Dilated causal 1-D conv for TCN residual blocks
  - [x] `auto_correlation_ptx` — FFT magnitude-squared step for Autoformer/TimesNet
  - [x] `revin_normalize_ptx` — RevIN normalise with per-(n,c) stats + learnable affine
  - [x] `multirate_pool_ptx` — Average pool at variable stride for NHiTS multi-rate sampling
  - [x] `period_detect_ptx` — Top-k FFT magnitude reduction for TimesNet period detection

- [x] **Normalisation** (norm/) — `RevIn` (reversible instance norm with forward+inverse, Bessel-corrected stats), `InstanceNorm1d` (per-variate instance norm with optional affine)

- [x] **Decomposition** (decomp/) — `MovingAvg` (centred, replicate-pad), `SeriesDecomp` (trend + seasonal split matching Autoformer)

- [x] **Patch embedding** (patch/) — `PatchEmbed1d` (overlapping 1-D patches, Xavier init, univariate + multivariate `forward_mv`)

- [x] **TCN** (tcn/) — `TcnBlock` (weight-normalised dilated causal conv, Kaiming He init, optional 1×1 residual projection), `TcnEncoder` (exponential dilation schedule 2^i, tiny/default configs)

- [x] **NHiTS** (nhits/) — `MultiRateSampler` (avg pool + nearest-neighbour upsample), `NHitsBlock` (pool→MLP→backcast+forecast heads), `NHits` (hierarchical residual stacks with pool_sizes=[1,2,4])

- [x] **PatchTST** (patchtst/) — `PatchTst` (channel-independent patches → sinusoidal PE → N×pre-LN TransformerLayer → per-variate linear head), `PatchTstConfig::tiny/base`

- [x] **TimesNet** (timesnet/) — `TimesBlock` (O(T²) DFT period detection → top-k 2-D reshape → depthwise 3×3 conv → weighted sum → residual + LN), `TimesNet` (input proj → blocks → flatten → linear head)

- [x] **iTransformer** (itransformer/) — `InvertedBlock` (attention over C variate tokens), `ITransformer` (variate embedding → N blocks → per-variate head), `ITransformerConfig::tiny/base`

- [x] **Forecasting heads** (head/) — `LinearHead` (in→out, batch + per-variate ts variants), `MlpHead` (in→hidden→out with ReLU, Kaiming init for layer 1)

- [x] **Integration tests** (lib.rs) — 20 E2E tests covering all modules + PTX × 6 SM versions
- [x] **Benchmarks** (benches/ts_ops.rs) — 7 PTX bench groups × 4 SM versions + 5 architecture forward benches

---

## Future Work / Beyond v1.0

### Performance & Optimization
- [x] Stream-K GEMM scheduling (Hopper+ load balancing)
- [x] Persistent kernel GEMM (SM occupancy optimization)
- [x] Warp-specialization for Hopper TMA (Tensor Memory Accelerator)
- [x] FP6 / FP4 mixed-precision training kernels (precision/fp4_fp6_ops.rs) -- Blackwell sub-byte GEMM with micro-scaling
- [x] Graph-based kernel fusion (CUDA Graph equivalent)
- [x] Cooperative groups API

### Multi-GPU & Distributed
- [x] Multi-GPU support with peer-to-peer memory access
- [x] NCCL-equivalent collective operations (AllReduce, AllGather, ReduceScatter) -- comm.rs in oxicuda umbrella
- [x] NVLink/NVSwitch topology-aware communication (nvlink_topology.rs) -- topology discovery, optimal ring/tree, Dijkstra routing
- [x] Multi-node distributed training support (distributed.rs) -- TcpStore/FileStore rendezvous, gradient bucketing, ZeRO sharding
- [x] Pipeline parallelism primitives (pipeline_parallel.rs) -- GPipe, 1F1B, interleaved, zero-bubble schedulers
- [x] Distributed inference engine (oxicuda-dist-infer Vol.12) -- TP/SP/EP parallelism, distributed KV cache, prefix-affinity routing

### Additional Backends
- [x] AMD ROCm backend (HIP runtime) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce)
- [x] Intel oneAPI / Level Zero backend -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via OpenCL SPIR-V)
- [x] Apple Metal backend (via metal-rs) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via MSL shaders)
- [x] WASM + WebGPU backend (oxionnx-web) -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via WGSL shaders)
- [x] Vulkan Compute backend -- memory + compute ops (GEMM, Conv2D, Attention, unary, binary, reduce via SPIR-V compute shaders)

---

## Backend Compute Operations [COMPLETE]

All 5 alternative GPU backend crates now have compute operations (GEMM, Conv2D, Attention, unary elementwise, binary elementwise, reduce) fully wired up instead of returning `Unsupported`.

### oxicuda-webgpu — WGSL Shader Dispatch via wgpu
- [x] GEMM compute shader (WGSL tiled matrix multiply)
- [x] Unary elementwise (relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu)
- [x] Binary elementwise (add, sub, mul, div, max, min, pow) with `binary_wgsl` generator
- [x] Reduction (sum, max, min, mean) with `reduction_wgsl` + `reduction_final_wgsl`
- [x] Conv2D forward (WGSL NCHW compute shader + CPU fallback)
- [x] Attention (WGSL scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_wgsl` shader + trait override with 3D dispatch)
- [x] FP16 GEMM (`gemm_wgsl_f16` shader with `enable f16` + `gemm_f16()` method)
- 86 tests passing

### oxicuda-metal — MSL Shader Dispatch via metal-rs
- [x] GEMM compute shader (MSL tiled matrix multiply)
- [x] Unary elementwise (relu, sigmoid, tanh, exp, log, sqrt, abs, neg, gelu, silu)
- [x] Binary elementwise (add, sub, mul, div, max, min, pow) with `binary_msl` generator
- [x] Reduction (sum, max, min, mean) with dedicated MSL max/min/mean shaders
- [x] Conv2D forward (MSL NCHW compute shader + CPU fallback)
- [x] Attention (MSL scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_msl` shader + trait override with 3D threadgroup dispatch)
- [x] FP16 GEMM (`gemm_msl_f16` shader with Metal `half` type + `gemm_f16()` method)
- 121 tests passing

### oxicuda-vulkan — SPIR-V Compute Shader Dispatch via ash
- [x] GEMM compute shader (SPIR-V tiled matrix multiply)
- [x] Unary elementwise (SPIR-V generator for all standard ops)
- [x] Binary elementwise (SPIR-V generator for all standard ops)
- [x] Reduction (SPIR-V generator for sum, max, min, mean)
- [x] Conv2D forward (SPIR-V NCHW compute shader + CPU fallback)
- [x] Attention (SPIR-V scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_compute_shader` SPIR-V + trait override with 3D dispatch)
- 66 tests passing

### oxicuda-rocm — HIP Kernel String Generators + Host-Side Dispatch
- [x] GEMM kernel (HIP tiled matrix multiply)
- [x] Unary elementwise (HIP kernel generator)
- [x] Binary elementwise (HIP kernel generator)
- [x] Reduction (HIP kernel generator for sum, max, min, mean)
- [x] Attention kernel (HIP fused attention)
- [x] Conv2D kernel (HIP convolution)
- [x] Batched GEMM (`batched_gemm_hip` kernel + trait override with CPU fallback)
- 56 tests passing

### oxicuda-levelzero — OpenCL SPIR-V Kernel Dispatch
- [x] GEMM kernel (OpenCL SPIR-V generator)
- [x] Unary elementwise (OpenCL SPIR-V generator)
- [x] Binary elementwise (OpenCL SPIR-V generator)
- [x] Reduction (OpenCL SPIR-V generator for sum, max, min, mean)
- [x] Conv2D forward (OpenCL SPIR-V NCHW compute shader + CPU fallback)
- [x] Attention (OpenCL SPIR-V scaled dot-product + stable softmax + causal masking)
- [x] Batched GEMM (`batched_gemm_compute_shader` OpenCL SPIR-V + trait override with 3D dispatch)
- 69 tests passing

---

### Deep Learning Extensions
- [x] FlashAttention-3 with Hopper-specific optimizations
- [x] Speculative decoding attention kernels
- [x] RNN / LSTM / GRU cells
- [x] Deformable convolution
- [x] Transposed convolution (conv/transpose_conv.rs)
- [x] Sparse attention patterns (sliding window, dilated)
- [x] INT4 / NF4 quantization (QLoRA support)
- [x] Dynamic batching / continuous batching (dynamic_batch.rs) -- ContinuousBatcher, paged KV, speculative decoding

### Scientific Computing Extensions
- [x] Multi-GPU FFT (multi_gpu.rs) -- 1D slab decomposition across multiple devices
- [x] Sparse eigenvalue solver (Lanczos, Arnoldi)
- [x] Tensor decomposition (tensor_decomp.rs) -- CP-ALS, Tucker HOSVD/HOOI, TT-SVD
- [x] ODE/PDE solver kernels (ode_pde.rs) -- Euler, RK4, RK45, implicit Euler, BDF2, heat/wave/Poisson/advection
- [x] Monte Carlo simulation primitives (monte_carlo.rs) -- MC integration, MCMC (MH, HMC), financial MC, variance reduction

### Ecosystem Integration
- [x] ComputeBackend trait for SciRS2
- [x] OxiCudaComputeBackend implementation
- [x] OxiONNX GPU inference backend (onnx_backend/) -- IR graph, op implementations, executor, planner, fusion, shape inference
- [x] ToRSh GPU backend (tensor_backend/) -- tensor, dtype, autograd, ops, optimizer, mixed precision
- [x] TrustformeRS Transformer GPU backend (transformer_backend/) -- KV-cache, attention, scheduler, speculative decoding, sampling, quantization
- [x] Benchmarks suite (criterion) with CI regression tracking -- oxicuda/benches/ (ptx_generation, autotune_search, fft_planning, blas_dispatch, backend_operations)
- [~] Published documentation on docs.rs (2026-05-01)
  - **Status:** `[package.metadata.docs.rs]` added to all 34 subcrate Cargo.toml files; docs build cleanly with `cargo doc --no-deps --all-features` (zero errors, zero warnings)
  - **Remaining:** Actual publication requires `cargo publish` — pending `/bump` flow

### Tooling
- [x] oxicuda-prof -- GPU profiling and tracing tool (profiling hooks implemented in oxicuda/profiling.rs)
- [x] oxicuda-debug kernel debugging (debug.rs) -- KernelDebugger, MemoryChecker, NanInfChecker, PTX instrumenter
- [x] Visual PTX explorer (tui_explorer.rs) -- CFG visualization, register lifetimes, instruction mix, PTX diff, complexity metrics
- [x] Automatic kernel fusion pass in PTX IR

---

## Aggregated Blueprint Quality Gates

Summary of quality gates from all 5 blueprint volumes. Each crate's TODO.md contains full details.

**Completed gates (2026-04-11):** A1, A3, A4 (autotune); P4, P7 (ptx); S15 (rand); plus NIST SP 800-22 statistical tests, FlashAttention-3 Hopper kernel bodies, multi-GPU FFT executor, matrix function PTX kernel bodies, backward error formula tests, sparse auto-selection heuristics.

**Completed gates (2026-04-11, Batch 15):** A1, A2, A3, A4 (autotune); P4, P7, wgmma, mma, WMMA, ldmatrix, cp.async, 3/4-stage pipeline, FP4 E2M1, tcgen05, TMA/cp.async.bulk, cluster barrier, griddepcontrol (ptx); D1-D4 docs; driver cluster launch, TMA descriptors, sm_100/120 occupancy, error injection; launch tracing; DNN FlashAttn tile, Winograd 4×, GQA, FP8 quant, accuracy suite.

### Vol.1 — Foundation (oxicuda-driver, oxicuda-memory, oxicuda-launch)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| F1–F5 | Dynamic load, multi-GPU, context, stream, E2E kernel | P0 | [ ] |
| F6–F8 | DeviceBuffer, PinnedBuffer, MemoryPool | P0 | [ ] |
| F9–F10 | Error handling, Drop resource release | P0 | [ ] |
| NF2 | H2D / D2H bandwidth | ≥ 95% of PCIe theoretical | [ ] |
| NF3 | Kernel launch overhead | < 1 μs above raw `cuLaunchKernel` | [ ] |
| NF4 | Memory leak detection | Zero leaks (`compute-sanitizer`) | [ ] |
| D1–D4 | Docs: README, architecture.md, API docs, examples | — | [x] |

### Vol.2 — PTX + Autotuner (oxicuda-ptx, oxicuda-autotune)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| P1–P8 | PTX generation: vector_add, GEMM sm_80, GEMM sm_90, elementwise, reduction, Tensor Core MMA, cache, ptxas compat | P0–P1 | [ ] |
| A1–A5 | Autotuner: pruning, stability, DB round-trip, dispatcher, CLI | P0–P1 | [ ] |
| A6 | Best autotuned config performance | ≥ 80% cuBLAS GEMM on A100 | [ ] |

### Vol.3 — BLAS (oxicuda-blas)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| G1–G9 | GEMM (F16/BF16/F32/F64/FP8), Batched GEMM, BLAS L1 | P0–P1 | [ ] |
| G10–G14 | BLAS L2/L3, elementwise, reduction, epilogue fusion | P0–P1 | [ ] |
| P1–P3 | GEMM F16/F32/F64 M=N=K=4096 | ≥ 95% cuBLAS | [ ] |
| P4 | Batched GEMM 1000×(256³) | ≥ 90% cuBLAS | [ ] |
| P5–P6 | Softmax 4096², axpy 10M elements | ≥ 90–95% cuDNN/cuBLAS | [ ] |

### Vol.4 — DNN (oxicuda-dnn)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| D1–D10 | Conv2D, FlashAttention, PagedAttention, RoPE, MoE routing + fused | P0–P1 | [ ] |
| D11–D20 | LayerNorm, RMSNorm, BatchNorm, GroupNorm, Pooling, Quantize, fused ops | P0–P2 | [ ] |
| P1 | Conv2D ResNet-50 layer3 | ≥ 90% cuDNN | [ ] |
| P2 | FlashAttention seq=2048, d=128, FP16 | ≥ 90% FlashAttention-2 | [ ] |
| P3 | PagedAttention decode batch=32, seq=4096 | ≥ 85% vLLM | [ ] |
| P4–P8 | MoE, LayerNorm, RMSNorm, BatchNorm, fused Conv+BN+ReLU | ≥ 90–95% / 2× speedup | [ ] |

### Vol.5 — Scientific Computing (oxicuda-fft/sparse/solver/rand + oxicuda umbrella)

| Gate | Requirement | Target | Status |
|------|-------------|--------|--------|
| S1–S6 | FFT 1D/2D/3D/batched/arbitrary-size | P0–P1 | [ ] |
| S7–S8 | SpMV / SpMM CSR correctness | P0 | [ ] |
| S9–S14 | LU, QR, Cholesky, SVD, Eigenvalue, Batched SVD | P0–P1 | [ ] |
| S15 | Philox uniform + normal correctness | P0 | [ ] |
| S16–S21 | SciRS2, oxionnx, ToRSh, TrustformeRS integration + CI/CD | P0–P1 | [ ] |
| P1–P2 | FFT 1D (2²⁰) and 2D (1024²) | ≥ 90% and ≥ 85% cuFFT | [ ] |
| P3 | SpMV CSR | ≥ 85% cuSPARSE | [ ] |
| P4–P6 | LU, SVD, Cholesky | ≥ 85–90% cuSOLVER | [ ] |
| P7 | Philox 100M samples | ≥ 95% cuRAND | [ ] |
| P8 | SciRS2 E2E typical workflow | ≥ 5× vs CPU-only SciRS2 | [ ] |

---

## Cross-Crate Numerical Accuracy Summary

| Library | Precision | Tolerance |
|---------|-----------|-----------|
| oxicuda-blas (GEMM) | FP64 | < 1e-14 |
| oxicuda-blas (GEMM) | FP32 | < 1e-5 |
| oxicuda-blas (GEMM) | FP16 | < 1e-2 |
| oxicuda-blas (GEMM) | BF16 | < 5e-2 |
| oxicuda-blas (GEMM) | FP8 | < 1e-1 |
| oxicuda-dnn (Conv2D) | FP16 / FP32 | < 1e-2 / < 1e-5 |
| oxicuda-dnn (FlashAttn) | FP16 / FP32 | < 2e-2 / < 1e-5 |
| oxicuda-dnn (Norms) | FP16 / FP32 | < 1e-3 / < 1e-6 |
| oxicuda-fft (round-trip) | FP32 | < N × 1.19e-7 |
| oxicuda-fft (round-trip) | FP64 | < N × 2.22e-16 |
| oxicuda-sparse (SpMV) | FP32 / FP64 | < 1e-5 / < 1e-14 |
| oxicuda-solver (residual) | FP32 / FP64 | < 1e-5 / < 1e-12 |

---

## v1.0 Completion Criteria (Vol.5 Sec 10.3)

| # | Condition | Status |
|---|-----------|--------|
| 1 | All SciRS2 CUDA dependencies eliminated (pure OxiCUDA backend) | [ ] Verify |
| 2 | oxionnx GPU inference operational on OxiCUDA backend | [ ] Verify |
| 3 | Major benchmarks achieve ≥ 95% of cuBLAS / cuDNN / cuFFT / cuSOLVER | [ ] Verify |
| 4 | Zero external dependencies beyond NVIDIA GPU driver (Pure Rust) | [ ] Verify |
| 5 | CI/CD pipeline with GPU tests + performance regression detection (5% threshold) | [ ] Verify |
| 6 | Documentation and examples cover all public API | [ ] Verify |

---

## Post-v1.0 Roadmap (from Blueprints)

| Version | Theme | Key Features | Status |
|---------|-------|-------------|--------|
| v1.1 | Multi-GPU | NCCL equivalent, NVLink topology, pipeline parallelism, distributed training | ✓ Done |
| v1.2 | Training | Gradient checkpointing, mixed-precision training, optimizer states on GPU | ✓ Done |
| v1.3 | Blackwell | FP4 / FP6 compute, 5th-gen Tensor Core, sm_100 / sm_120 optimized paths | ✓ Done |
| v2.0 | AMD ROCm | HIP backend, same API surface, ROCm 5.x+ | ✓ Done |
| v2.1 | Intel oneAPI | SYCL backend, Intel GPU (Arc, Ponte Vecchio) | ✓ Done |
| v3.0 | WASM + WebGPU | Browser GPU compute via WebGPU API | ✓ Done |
| v3.1 | State Space Models | S4 / Mamba / Mamba-2 (SSD) / RWKV — linear-time attention alternatives | ✓ Done |

---

## Maintenance

- [x] preemptive-splitrs-near-cap (planned 2026-05-01)
  - **Goal:** Prevent three files from crossing the 2000-line refactoring cap; split now while seams are clean
  - **Design:** Priority order: (1) `crates/oxicuda-sparse/src/ops/batched.rs` (1950 LoC, 665 test lines) — extract test module; (2) `crates/oxicuda/src/tensor_backend/ops.rs` (1986 LoC, 316-line test block) — extract test module; (3) `crates/oxicuda-blas/src/precision/fp4_fp6_ops.rs` (1955 LoC) — split along FP6/FP4 banner seam; use `splitrs` CLI for each; run clippy -D warnings immediately after each split
  - **Files:** Driven by `splitrs` output; verify with `rslines 50` post-split
  - **Tests:** Existing tests must pass unchanged after each split; `cargo nextest run -p <crate> --all-features`
  - **Risk:** Low for test-module extractions (prefer `mod tests { use super::*; }` pattern); medium for fp4_fp6_ops.rs if internal helpers need `pub(crate)` widening
