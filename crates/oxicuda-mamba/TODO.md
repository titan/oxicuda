# oxicuda-mamba TODO

Pure-Rust State Space Model (SSM) primitives for OxiCUDA: S4 (HiPPO-LegS / DPLR),
Mamba selective scan (S6), Mamba-2 (SSD), and RWKV time-mixing -- linear-time
alternatives to attention. Part of [OxiCUDA](https://github.com/cool-japan/oxicuda)
(Vol.19).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 10,396 (24 files, Rust 7,175 code + 1,870 comments + 1,351 blanks)
- **Tests:** 339 passing (#[test] count in src/)
- **Crate:** `oxicuda-mamba` -- Vol.19 State Space Model Primitives

### Completed [x]

#### Core Infrastructure
- [x] `error.rs` -- `MambaError` (15 variants): `DimensionMismatch`, `ShapeMismatch`,
      `EmptyInput`, `InvalidSeqLen`, `InvalidSsmOrder`, `InvalidModelDim`,
      `NonPositiveDelta`, `InvalidChunkSize`, `HeadDimMismatch`, `WeightShapeMismatch`,
      `NonFinite`, `Internal`; `MambaResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (Box-Muller normals, Fisher-Yates shuffle),
      `MambaHandle`
- [x] `lib.rs` -- crate root with `prelude` module and 20 E2E integration tests

#### PTX Kernels (`ptx_kernels.rs`, 7 kernels x 6 SM versions: 75/80/86/90/100/120)
- [x] `selective_scan_ptx` -- Mamba S6: `h = A_bar * h + B_bar * u`, `y = C * h`;
      per-channel sequential recurrence
- [x] `parallel_scan_ptx` -- warp-level `(A, b)` associative prefix scan via
      `shfl.sync.down.b32` butterfly
- [x] `depthwise_conv1d_ptx` -- causal 1-D depthwise conv with zero-pad, `fma.rn.f32`
- [x] `wkv_forward_ptx` -- RWKV WKV with numerically-stable running-max pivot;
      `ex2.approx.f32`
- [x] `ssd_chunk_ptx` -- Mamba-2 SSD chunk: causal `prod A_k` accumulation per output
      position
- [x] `hippo_legendre_ptx` -- HiPPO-LegS forward Euler:
      `c_n' = c_n*(1 - delta*(n+1)) + delta*sqrt(2n+1)*u`
- [x] `rms_norm_silu_ptx` -- fused RMSNorm + SiLU gate; warp butterfly sum via
      `shfl.sync.bfly.b32`

#### SSM Core (`ssm/`, 3 files + mod)
- [x] `ssm/discretize.rs` -- ZOH (`A_bar = exp(delta * A)`), Bilinear (Tustin), Euler;
      L'Hopital limit for `|A| ~= 0`
- [x] `ssm/parallel_scan.rs` -- `ScanPair {a, b}` with associative combine operator,
      inclusive / exclusive prefix scan, `ssm_state_scan(a_bar, b_bar_u)`
- [x] `ssm/ssm_kernel.rs` -- `SsmKernel`: batch-aware
      `h[b, t, d, n] = A_bar * h_prev + B_bar * u` recurrence, ZOH discretization,
      output `y = sum C * h`

#### S4 Architecture (`s4/`, 3 files + mod)
- [x] `s4/hippo.rs` -- `hippo_legs(n)`: HiPPO-LegS A matrix (lower-triangular,
      `A[n,k] = -sqrt(2n+1)*sqrt(2k+1)`) and B vector; `hippo_legs_diag`;
      `hippo_nplr` NPLR decomposition (`lambda[n] = -(n+0.5)`,
      `p = q = sqrt(n+0.5)`)
- [x] `s4/dplr.rs` -- `Dplr {lambda, p, q}`: `A = diag(lambda) - p * q^T`;
      `from_hippo`, `to_dense`, ZOH SSM kernel computation via mode decomposition
- [x] `s4/s4_layer.rs` -- `S4Layer`: multi-channel convolutional mode, `naive_conv1d`
      O(L^2) reference, optional bidirectional averaging, `S4Config` builder

#### Mamba S6 (`mamba/`, 3 files + mod)
- [x] `mamba/selective_scan.rs` -- `selective_scan`: input-dependent
      `delta = softplus(proj)`, `A_bar = exp(delta tensor A)`,
      `B_bar = delta tensor B_proj`, sequential state recurrence; `softplus`
      with +/-20 stability clamp
- [x] `mamba/mamba_block.rs` -- `MambaBlock`: in_proj -> x / z split ->
      conv1d + SiLU -> selective_scan -> D skip -> SiLU gate -> out_proj + residual;
      includes `rms_norm`, `linear`, `silu`, `causal_depthwise_conv1d` helpers
- [x] `mamba/mamba_model.rs` -- `MambaModel`: TokenEmbedding -> N x MambaBlock ->
      RMSNorm -> LM head; `forward` returns logits, `next_token` greedy decode;
      `MambaConfig::tiny()` test preset

#### Mamba-2 / SSD (`mamba2/`, 3 files + mod)
- [x] `mamba2/ssd.rs` -- `ssd_naive` O(L^2 * N) semi-separable matrix-vector product;
      `ssd_recurrent` O(L * N) state form; `verify_ssd_equivalence` agreement check
      (tol 1e-4)
- [x] `mamba2/chunk_scan.rs` -- `ChunkConfig` with ceiling-division chunks;
      `chunk_scan`: intra-chunk naive SSD + inter-chunk boundary state propagation;
      `verify_chunk_equivalence`
- [x] `mamba2/mamba2_block.rs` -- `Mamba2Block`: multi-head SSD with
      `a[t] = sigmoid(-exp(a_h))`, per-head `chunk_scan`, D skip, RMSNorm,
      out_proj + residual

#### RWKV (`rwkv/`, 3 files + mod)
- [x] `rwkv/time_mixing.rs` -- `WkvState {a, b, p}` recurrent state;
      numerically-stable WKV via running-max pivot; `TimeMixingLayer` full RWKV-4
      pipeline: LN -> token-shift -> r/k/v projection -> WKV -> sigmoid gate ->
      output projection; `layer_norm`, `sigmoid` helpers
- [x] `rwkv/channel_mixing.rs` -- `ChannelMixingLayer`: token-shift ->
      sigmoid-gated receptance -> Square-ReLU expansion -> value contraction;
      `square_relu(x) = max(0, x)^2`
- [x] `rwkv/rwkv_block.rs` -- `RwkvBlock`: pre-norm residual:
      `y = x + time_mixing(LN_1(x))`, `y = y + channel_mixing(LN_2(y))`

#### Integration tests (`lib.rs::tests`)
- [x] 20 E2E tests covering ZOH/Bilinear/Euler discretization, parallel-scan
      associativity, S4 HiPPO + DPLR + S4Layer convolution, Mamba selective scan +
      block + model greedy decode, Mamba-2 SSD equivalence + chunk-scan, RWKV WKV
      stability + time-mixing + channel-mixing + block, plus PTX generation across
      6 SM versions

### Future Enhancements [ ]

#### P0 -- Critical (Mainstream Coverage / Correctness)
- [ ] Selective-scan parallel (Blelloch) kernel exposed end-to-end through `MambaBlock`
      (currently CPU sequential reference + PTX template only)
- [ ] FP16 / BF16 mixed-precision selective-scan (FP32 accumulation for `h`)
- [ ] Mamba-2 SSD chunk-scan GPU dispatch via `ssd_chunk_ptx`
- [x] RWKV-5 / RWKV-6 time-mixing variants
- [ ] Backwards pass for `selective_scan` (training support, not just inference)

#### P1 -- Important (Architecture and Feature Coverage)
- [x] S5 / Liquid-S4 variants (closed-form HiPPO transitions)
- [x] Bidirectional SSM helper layer for sequence-classification tasks (bidirectional_ssm.rs -- diagonal-state SSM forward + reverse scan, Sum or Concat combination for sequence-classification)
- [x] FFT-based S4 convolutional mode (O(L log L) replacement for `naive_conv1d`) (s4/s4_fft.rs -- radix-2 Cooley-Tukey FFT, fft_conv1d matching naive_conv1d, causal O(L log L) s4_fft_conv)
- [x] HiPPO-LegT / HiPPO-FOUT alternative HiPPO matrices
- [x] Mamba MoE (Mixture-of-Experts) sparse routing layer (mamba_moe.rs -- per-token router top-k expert SSM blocks, renormalized softmax weighted sum, load-balance loss N·mean(f·P) for uniform usage)
- [ ] Hybrid Mamba-Attention block (e.g. Jamba-style interleaving)

#### P2 -- Nice-to-Have (Research / Advanced)
- [ ] Quantised Mamba (Q-Mamba, INT8 / INT4 inference)
- [x] xLSTM (sLSTM + mLSTM) experimental layer
- [x] Hyena hierarchy long-conv layer (hyena.rs -- Poli 2023; implicit positional-MLP long-conv filter + multiplicative gating recurrence of order steps, reuses FFT conv)
- [ ] CUDA-Graph capture of multi-layer Mamba forward for inference
- [ ] Distributed multi-GPU Mamba (sequence-parallel across long contexts)
- [ ] State-checkpointing helper for long-context inference (kv-cache analogue)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmarking harness | Yes |

No CUDA SDK, no C/Fortran. PTX is emitted as Rust string templates and executed
through the oxicuda-driver runtime loader.

## Quality Status

- Warnings: 0 (clippy clean, no_warnings policy)
- Tests: 339 passing
- unwrap() calls: 0 in production code (no-unwrap policy)
- Files under 2000 SLoC: All
- Pure-Rust default features: Yes (Pure Rust Policy)

## Performance Targets

SSMs are sequential (Mamba S6) or low-rank-convolutional (S4) -- targets reflect
both regimes:

| Kernel | Sizes | Priority |
|--------|-------|----------|
| `selective_scan_ptx` | seq_len in {1K, 4K, 16K}, d_state in {16, 64}, channels in {1K, 4K} | P0 |
| `parallel_scan_ptx` | seq_len up to 64K (warp granularity) | P0 |
| `depthwise_conv1d_ptx` | causal kernel 4, channels 1K -- 4K, seq 1K -- 16K | P0 |
| `ssd_chunk_ptx` | chunk in {64, 128, 256}, heads 8 -- 32 | P1 |
| `wkv_forward_ptx` | embed 1K -- 4K, seq 1K -- 16K | P1 |
| `hippo_legendre_ptx` | n_state 64 -- 256, seq 1K -- 16K | P2 |
| `rms_norm_silu_ptx` | last-dim 1K -- 4K | P2 |

Target: Mamba selective-scan within 30% of HF `mamba_ssm` reference (parallel kernel
required); RWKV WKV within 20% of `rwkv_kernel` reference.

## Notes

- Selective-scan in `mamba/selective_scan.rs` is the CPU sequential reference; the
  GPU parallel kernel is generated by `parallel_scan_ptx`
- `softplus` clamp at +/-20 prevents `exp` overflow / underflow for `delta`
- WKV recurrence uses running-max pivot for numerical stability
  (paper-faithful RWKV-4)
- SSD intra-chunk path is O(L^2 * N); inter-chunk boundary state propagation is O(L)
- `MambaConfig::tiny()` and `Mamba2Block::tiny()` provide reproducible test presets
- macOS: kernels compile to PTX strings but device launch returns `UnsupportedPlatform`

---

## Architecture-Specific Deepening

### Ampere (sm_80) / Ada (sm_89)
- [x] `parallel_scan_ptx` warp-shuffle butterfly via `shfl.sync.down.b32`
- [x] `rms_norm_silu_ptx` warp-shuffle butterfly via `shfl.sync.bfly.b32`
- [x] `wkv_forward_ptx` uses `ex2.approx.f32` HW path
- [x] PTX × SM 80, 86 generation verified in integration tests
- [ ] `cp.async` double-buffer for selective-scan parameter staging
- [ ] FP16 selective-scan with FP32 `h` accumulator

### Hopper (sm_90 / sm_90a)
- [x] PTX SM 90 emission tested for all 7 kernels
- [ ] TMA (`cp.async.bulk`) for long-context parameter tiles
- [ ] Distributed-shared-memory for SSD inter-chunk boundary propagation
- [ ] `wgmma.mma_async` for SSD chunk dense path
- [ ] Cluster-launch variant for very long sequences (>=64K)

### Blackwell (sm_100 / sm_120)
- [x] PTX SM 100 / 120 emission tested
- [ ] FP8 (E4M3) Mamba inference path
- [ ] Tensor-Memory (TMEM) staging for selective-scan parameters

---

## Deepening Opportunities

> Items marked `[x]` represent API surface coverage. The items below represent the
> gap between the current implementation depth and blueprint-grade depth.

### Test Coverage
- [x] Discretization: ZOH / Bilinear / Euler agreement on small A
- [x] Parallel-scan associativity (left vs right combine) verified
- [x] S4 HiPPO-LegS shape + recurrence equivalence to NPLR
- [x] S4 DPLR mode-decomposition reconstruction within 1e-5
- [x] Selective-scan softplus clamp prevents NaN at extreme inputs
- [x] Mamba block / model forward shape and finiteness
- [x] Mamba-2 `verify_ssd_equivalence` naive vs recurrent within 1e-4
- [x] Mamba-2 `verify_chunk_equivalence` intra-chunk vs full within 1e-4
- [x] RWKV WKV stability under large-positive-key sequences (no NaN/Inf)
- [x] RWKV channel-mixing Square-ReLU monotonicity test
- [x] PTX generation across 6 SM versions: 75 / 80 / 86 / 90 / 100 / 120
- [ ] GPU-hardware correctness for all 7 kernels (gated behind `gpu-tests`)
- [ ] Numerical agreement with `mamba-ssm` reference within 1e-3 relative
- [ ] Mamba LM head perplexity match on small reference dataset (e.g. WikiText-2)
- [ ] RWKV-4 generation match vs reference HF model on first 50 tokens

### Implementation Deepening
- [ ] End-to-end `MambaModel::forward` GPU dispatch (currently CPU sequential)
- [ ] Backward pass for selective-scan and SSD (training-capable)
- [ ] `S4Layer` FFT-based convolution mode (link with `oxicuda-fft`)
- [ ] State-checkpointing helper for streaming long-context inference
- [ ] Multi-GPU sequence-parallel Mamba (split L across devices, exchange boundary `h`)

### Benchmark Coverage
- [x] `benches/mamba_ops.rs` Criterion harness wired (CPU-side PTX generation +
      scan / block forward)
- [ ] GPU-side throughput vs reference (`mamba_ssm`, `rwkv-cuda`) once Linux+NVIDIA
      harness is available
- [ ] Long-context sweep (L in 1K / 4K / 16K / 64K) for selective-scan throughput
