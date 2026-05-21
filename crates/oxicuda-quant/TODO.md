# oxicuda-quant TODO

GPU-accelerated quantization and model compression engine: post-training quantization (PTQ), quantization-aware training (QAT), pruning, knowledge distillation, and mixed-precision sensitivity analysis. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.10).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 5,887 SLoC across 24 files (includes Markdown doc-comments) / 4,122 pure Rust SLoC**

Comprehensive PTQ + QAT + Pruning + Distillation + Mixed-Precision Analysis suite for LLM and DNN
deployment. All quantization schemes commonly used in production inference frameworks (MinMax,
NF4 QLoRA, FP8 E4M3/E5M2, GPTQ Hessian-OBC, SmoothQuant) are implemented.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- 12 `QuantError` variants: DimensionMismatch, EmptyInput, InvalidScale, InvalidBitWidth, GroupSizeMismatch, CalibrationRequired, SingularHessian, TeacherStudentMismatch, AllZeroPruning, NonFiniteFp8, InfeasibleCompressionTarget, InvalidConfig
- [x] `lib.rs` -- module declarations and top-level `QuantError`/`QuantResult` re-exports

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 5 GPU-side quantization kernels (16.7 KB of PTX generators)
  - `fake_quant_ptx` -- STE-aware fake quantization for QAT
  - `int8_quant_ptx` / `int8_dequant_ptx` -- INT8 quant/dequant with scale + zero-point
  - `nf4_dequant_ptx` -- NF4 lookup table in shared memory
  - `prune_mask_ptx` -- apply sparsity mask in-place

#### Quantization Schemes (`scheme/`)
- [x] `scheme/mod.rs` -- module organization
- [x] `scheme/minmax.rs` -- `MinMaxQuantizer` -- INT4/INT8 Symmetric/Asymmetric, PerTensor/PerChannel/PerGroup granularity
- [x] `scheme/nf4.rs` -- `Nf4Quantizer` -- QLoRA NF4 with exact quantile lookup, nibble packing, absmax blocks
- [x] `scheme/fp8.rs` -- `Fp8Codec` -- E4M3 (max=448) and E5M2 (max=57344) via IEEE 754 bit manipulation
- [x] `scheme/gptq.rs` -- `GptqQuantizer` -- Hessian-based OBC via Cholesky + L^-1, column-wise weight correction
- [x] `scheme/smooth_quant.rs` -- `SmoothQuantMigrator` -- alpha-scaled activation/weight migration preserving output

#### QAT (`qat/`)
- [x] `qat/mod.rs` -- module organization
- [x] `qat/fake_quant.rs` -- `FakeQuantize` -- quantize -> dequantize forward, STE backward; enable/disable mode
- [x] `qat/observer.rs` -- three observer types
  - `MinMaxObserver` -- running global min/max, compute scale/zp
  - `MovingAvgObserver` -- EMA momentum update of min/max
  - `HistogramObserver` -- histogram + min-MSE percentile clipping search

#### Pruning (`pruning/`)
- [x] `pruning/mod.rs` -- module organization
- [x] `pruning/mask.rs` -- `SparseMask`: boolean weight mask; `sparsity()`, `apply()`, `apply_in_place()`, `and`/`or` compose
- [x] `pruning/magnitude.rs` -- `MagnitudePruner`: L1/L2 unstructured with grouped variant
- [x] `pruning/structured.rs` -- `StructuredPruner`: channel/filter/head granularity, L2-norm unit ranking

#### Knowledge Distillation (`distill/`)
- [x] `distill/mod.rs` -- module organization
- [x] `distill/loss.rs` -- `DistilLoss` enum -- KL (tau^2-scaled), MSE, cosine, combined
- [x] `distill/response.rs` -- `ResponseDistiller` -- soft + hard label combination, batch loss
- [x] `distill/feature.rs` -- `FeatureDistiller` -- per-layer weighted feature matching, `normalise_weights`

#### Compression Analysis (`analysis/`)
- [x] `analysis/mod.rs` -- module organization
- [x] `analysis/sensitivity.rs` -- `SensitivityAnalyzer`: per-layer MSE across bit-widths via MinMax symmetric
- [x] `analysis/metrics.rs` -- `CompressionMetrics` + `ModelCompressionMetrics`: bits, ratio, sparsity, weighted MSE
- [x] `analysis/policy.rs` -- `MixedPrecisionPolicy`: greedy sensitivity-guided bit assignment

### Future Enhancements

#### P0 -- Critical (PTQ Coverage)
- [x] INT4/INT8 MinMax with Per-Group granularity (`scheme/minmax.rs`)
- [x] NF4 (QLoRA) lookup table + nibble packing (`scheme/nf4.rs`)
- [x] FP8 E4M3/E5M2 IEEE 754 codecs (`scheme/fp8.rs`)
- [x] GPTQ Hessian-OBC weight correction (`scheme/gptq.rs`)

#### P1 -- Important (QAT + Distillation)
- [x] STE-backed `FakeQuantize` (`qat/fake_quant.rs`)
- [x] Three observer types (MinMax / MovingAvg / Histogram) (`qat/observer.rs`)
- [x] KL / MSE / cosine distillation losses (`distill/loss.rs`)
- [x] SmoothQuant alpha migration (`scheme/smooth_quant.rs`)

#### P2 -- Nice-to-Have (Structured / Mixed-Precision)
- [x] Magnitude pruning (L1/L2 unstructured + grouped) (`pruning/magnitude.rs`)
- [x] Structured pruning (channel/filter/head) (`pruning/structured.rs`)
- [x] Sensitivity-guided greedy bit assignment (`analysis/policy.rs`)
- [ ] (P2) AWQ activation-aware quantization (no current implementation; could extend `scheme/`)
- [ ] (P2) GGUF/AutoGPTQ container import/export (requires `oxicuda-arc` integration)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| oxicuda-driver | CUDA Driver API wrapper (libloading) | Yes (runtime FFI only) |
| thiserror | Error derive macros | Yes |
| tracing | Structured logging for calibration runs | Yes |
| num-traits | Numeric trait bounds | Yes |

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 151 passing (root TODO.md count)
- unwrap() calls: 0 (production code)
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS: compiles, returns `UnsupportedPlatform` at runtime

## Performance Targets

| Operation | Target |
|-----------|--------|
| `int8_quant_ptx` -- 100M weights | >= 90% bandwidth-limited peak (sm_80+) |
| `nf4_dequant_ptx` -- 7B weight load | >= 80% bandwidth-limited peak |
| GPTQ Cholesky -- 4096x4096 layer | < 5 s on sm_80 |
| MinMax PerChannel calibration | 1 GFLOPS on 1B-element calibration set |
| Mixed-precision policy search | < 1 s for 32-layer model, 4 bit-width options |

## Numerical Accuracy Requirements

| Scheme | Tolerance vs FP32 reference |
|--------|-----------------------------|
| INT8 symmetric PerTensor | rel < 1e-2 average across tensor |
| INT4 symmetric PerGroup-128 | rel < 5e-2 average across tensor |
| NF4 absmax-block | rel < 8e-2 (QLoRA target) |
| FP8 E4M3 (max=448) | rel < 5e-3 in [-256, 256] range |
| FP8 E5M2 (max=57344) | rel < 2e-2 in [-32768, 32768] range |
| GPTQ post-OBC reconstruction | rel < 1.5e-2 on calibration set |

## Architecture-Specific Deepening Opportunities

### Hopper (sm_90 / sm_90a)
- [x] FP8 E4M3/E5M2 codecs ready for `wgmma` tensor-core integration via `oxicuda-blas`
- [ ] Tensor-core fused dequant -> GEMM path on sm_90 (deferred -- requires GPU verification)

### Ampere (sm_80 / sm_86 / sm_89)
- [x] INT8 quantization compatible with `mma.sync.aligned.m16n8k32.s8` tensor cores
- [x] NF4 codec compatible with INT4 tensor-core path on sm_80+
- [ ] cp.async-driven calibration data streaming (deferred)

### Ada (sm_89)
- [x] FP8 codecs verified to match Ada Lovelace MMA instruction expectations
- [ ] End-to-end QLoRA fine-tuning on Ada FP8 GEMM (requires hardware run)

## Deepening Opportunities

### Verification Gaps
- [x] All 5 PTQ schemes covered by unit tests with reference FP32 baseline
- [x] All 3 QAT observers exercise calibration loop in dedicated tests
- [x] Sensitivity analyzer cross-checked against `compute_metrics` ground truth
- [ ] GPTQ Hessian conditioning checks against pathological inputs (singular Hessian fallback exists but limited stress test)
- [ ] SmoothQuant alpha sweep across [0, 1] verified to preserve output identity exactly

### Implementation Deepening
- [x] PerGroup quantization with arbitrary group size in `MinMaxQuantizer`
- [x] Distillation `combined` loss supports user-provided weight tuple
- [x] Mixed-precision policy returns explicit per-layer bit assignment plan
- [ ] Sparsity-aware GPTQ (combine pruning mask with column-wise weight correction)
- [ ] AdaRound-style rounding optimization for INT8/INT4 minmax baseline

## Notes

- All quantization schemes are tensor-shape agnostic; calling code provides flat `&[f32]` slices.
- `ptx_kernels.rs` emits standard PTX; the same code generators are reused by `oxicuda-blas`'s quantized GEMM path.
- Benchmarks live in `benches/quant_ops.rs` (Criterion harness) -- CPU-side codec correctness only; GPU benchmarking awaits Linux+NVIDIA hardware.
- Future work integrating with `oxicuda-train` for end-to-end QAT training loops is tracked in the root TODO.md.
