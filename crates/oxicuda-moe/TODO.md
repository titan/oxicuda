# oxicuda-moe TODO

Mixture of Experts (MoE) primitives for OxiCUDA (Switch Transformer, GShard top-K routing, Expert Choice, Soft MoE). Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.35).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 9,382 SLoC (37 source files + 1 benches file) -- Coverage: full router family + expert FFN + auxiliary losses + complete MoE layer**

Current implementation covers Switch Transformer top-1 routing with capacity buffers and overflow token dropping, GShard-style top-K gating (softmax over experts, partial-sort top-k, optional Gaussian noise jitter via Box-Muller), Expert Choice routing (experts select preferred tokens for guaranteed load balance), Soft MoE (differentiable slot routing `D = softmax(X * Phi / sqrt(d))`, slot-aggregated expert inputs), GELU / SiLU / ReLU expert FFNs with Xavier init, SwiGLU expert `(SiLU(W1*x) (cdot) (W3*x)) * W2`, `ExpertBank` and `SwiGluBank` dispatch utilities, Switch load-balance loss `L_aux = n_e * sum f_i * P_i`, router z-loss `log^2(logsumexp(logits))`, routing entropy, expert utilization metrics, and a full `MoeLayer` combining router + expert bank + auxiliary losses.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `MoeError` (14 variants: DimensionMismatch, EmptyInput, InvalidExpertCount, InvalidTopK, InvalidCapacityFactor, ExpertIndexOutOfRange, NanEncountered, InvalidHiddenDim, InvalidInputDim, DispatchFailed, RouterNotInitialized, ExpertFfnError, SlotAssignmentError, Internal), `MoeResult<T>`
- [x] `handle.rs` -- `SmVersion` (Sm75/80/86/90/100/120), `LcgRng` (Knuth MMIX 64-bit LCG), `MoeHandle::default_handle()` (Sm80, device 0, seed 42)

#### Routing
- [x] `routing/top_k.rs` -- `TopKRouter`, `TopKConfig`, `TopKResult`, `topk()`; k=1 argmax, k=2 one-pass max-2, k>=3 partial sort; optional Gaussian noise jitter `N(0, sigma^2)` via Box-Muller; top-k score normalization
- [x] `routing/switch.rs` -- `switch_dispatch`, `switch_combine`, `SwitchDispatch`; capacity = `ceil(T / E * cap_factor)`, overflow tracking, combine via gate scores
- [x] `routing/expert_choice.rs` -- `expert_choice_route`, `expert_choice_combine`, `ExpertChoiceResult`; experts select top-c tokens from their score column for guaranteed load balance
- [x] `routing/soft_moe.rs` -- `SoftMoeRouter`; `dispatch_weights()` returns `[T, E*S]`, `aggregate_inputs()` (weighted average per slot), `combine_outputs()` (scatter back to tokens)

#### Expert FFN
- [x] `expert/ffn.rs` -- `ExpertFfn`, `ExpertActivation` (GELU / SiLU / ReLU), `SwiGluExpert` `(SiLU(W1 * x) (cdot) (W3 * x)) * W2`; Xavier init; batch forward
- [x] `expert/bank.rs` -- `ExpertBank`, `SwiGluBank`; N-expert bank; `forward_expert(idx, tokens)`, `forward_dispatched(x, assignments, scores)`

#### Losses
- [x] `loss/load_balance.rs` -- `load_balance_loss`, `compute_load_stats`, `LoadStats`; `L_aux = n_e * sum f_i * P_i`; per-expert fraction and mean probability
- [x] `loss/router_z.rs` -- `router_z_loss`; `(1/B) * sum_b log^2(LSE_b)` with stable logsumexp
- [x] `loss/entropy.rs` -- `routing_entropy`; `-(1/T) * sum_t sum_e p_{t,e} * log(p_{t,e} + eps)`

#### Metrics
- [x] `metrics/utilization.rs` -- `ExpertUtilization { tokens_per_expert, overflow_count, load_imbalance_ratio, utilization_fraction }`; `compute_utilization()`

#### MoeLayer
- [x] `layer/moe_layer.rs` -- `MoeLayer { router, experts }` with `MoeLayerConfig` and `MoeLayerOutput { hidden, aux_loss, n_overflows, load_stats }`; full forward path: route -> Switch dispatch -> expert bank -> combine -> auxiliary losses

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 7 GPU kernels x 6 SM versions (75/80/86/90/100/120):
  - [x] `top_k_gate_kernel` -- softmax + top-k selection per token
  - [x] `expert_dispatch_kernel` -- capacity-bounded token-to-expert slot assignment
  - [x] `expert_ffn_kernel` -- batched `y = W2 * GELU(W1 * x + b1) + b2` per token
  - [x] `expert_combine_kernel` -- weighted sum of expert outputs by gate scores
  - [x] `load_balance_loss_kernel` -- `n_e * sum f_i * P_i` reduction
  - [x] `router_z_loss_kernel` -- `log^2(logsumexp(logits))` per token, then mean reduction
  - [x] `soft_moe_dispatch_kernel` -- slot dispatch `D[t, s] = softmax(x * Phi / sqrt(d))`

#### Integration Tests
- [x] 12 e2e tests (lib.rs): top-k scores sum to 1, indices valid, Switch capacity respected, overflow counted, ExpertFfn finite output, ExpertFfn output shape preserved, SwiGLU finite, load-balance loss non-negative, router z-loss non-negative, Soft MoE dispatch sums to 1, MoeLayer forward shape, PTX kernels x 6 SM versions

#### Benchmarks
- [x] `benches/moe_ops.rs` -- 7 PTX kernel groups x 4 SM versions plus 6 algorithm benches: topk_routing_512tok_8exp_d256, expert_ffn_batch64, switch_dispatch_512tok, load_balance_512tok, moe_layer_128tok

### Future Enhancements

#### P0 -- Critical Algorithmic Coverage
- [x] Expert dropout (random expert masking during training) for robustness
- [x] Differentiable top-k via Gumbel-softmax for end-to-end gradient flow
- [x] Hash routing (deterministic token-to-expert mapping for cache-friendly serving)
- [x] All-to-all expert dispatch primitive for distributed multi-device MoE

#### P1 -- Important Features
- [x] BASE (Lewis et al. 2021) -- balanced assignment via Sinkhorn iterations
- [x] Stable MoE -- router stability tricks (auxiliary z-loss + sigmoid gating + load-balance loss combined)
- [ ] Sparse upcycling -- initialize MoE FFN weights from dense FFN checkpoint
- [x] Expert parallelism placement -- pinning experts to device groups
- [x] Megablocks-style block-sparse dispatched GEMM (avoid padding to capacity)
- [x] Multi-gate MoE (separate router per task) for multi-task learning

#### P2 -- Advanced / Research
- [ ] MoE-Mamba / MoE-State-Space routing
- [ ] Conditional computation routing (skip computation entirely for some tokens)
- [ ] Expert pruning / merging at inference (knowledge distillation)
- [ ] Layer-conditional routing (router shared / not shared across layers)
- [ ] Hierarchical routing (cluster experts into groups, route to group first then expert)
- [ ] Differentiable expert capacity (learnable per-expert capacity scale)
- [ ] `moe/mixtral.rs` — Mixtral-style sparse MoE (Jiang 2024): top-2 expert routing per token; expert-parallel sharding; gating network with aux-loss for load balance; `MixtralMoeLayer { n_experts, top_k: 2 }`
- [x] `moe/deepseek_moe.rs` — DeepSeekMoE (Dai 2024): fine-grained expert segmentation + shared experts; each token activates mₛ shared + mᵣ routed experts; `DeepSeekMoeConfig { n_shared, n_routed, top_k_routed }`
- [ ] `routing/expert_choice.rs` — Expert Choice (Zhou 2022): experts choose top-k tokens rather than tokens choosing experts; guaranteed perfect load balance; `ExpertChoiceRouter { capacity_factor: f32 }`
- [ ] `moe/lora_moe.rs` — LoRAMoE (Sheng 2024): mixture of LoRA adapters as experts; gating selects which LoRA module per token; continual learning via world knowledge preservation constraint

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none) | Standalone primitives crate | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

## Quality Status

- Tests: 303 passing (12 e2e in lib.rs + module unit tests)
- All production code uses `Result` / `Option` (no `unwrap()` outside tests)
- `clippy::all` warnings: 0
- `missing_docs` warnings: 0
- Files: 37 source `.rs` files, all under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns `UnsupportedPlatform` at runtime

## Performance Targets

Representative shapes for typical Mixture-of-Experts transformers.

| Operation | Configuration | Priority |
|-----------|---------------|----------|
| `top_k_gate_kernel` | 512 tokens, 8 experts, d_model 256 | P0 |
| `expert_dispatch_kernel` | 512 tokens, 8 experts, cap_factor 1.25 | P0 |
| `expert_ffn_kernel` | batch 64, d_model 256, ffn 1024 | P0 |
| `expert_combine_kernel` | 512 tokens, top-k 2 | P0 |
| `soft_moe_dispatch_kernel` | 128 tokens, 4 experts, 2 slots each | P1 |
| `load_balance_loss_kernel` | 512 tokens x 8 experts | P1 |
| `router_z_loss_kernel` | 512 tokens x 8 experts | P1 |

Target: MoE layer forward latency comparable to PyTorch + Megablocks reference for `top-1` Switch and `top-2` GShard configurations on `sm_80+`.

## Estimation vs Actual

| Metric | Description | Actual |
|--------|-------------|--------|
| Files | source `.rs` files under `src/` | 37 |
| SLoC | code lines (tokei) | ~9,382 |
| Tests | e2e + unit | 303 |
| Coverage | router algorithms | 4 (TopK, Switch, ExpertChoice, SoftMoE) |
| Coverage | expert types | 4 (GELU, SiLU, ReLU, SwiGLU) |

The current implementation provides a compact reference covering all four canonical MoE routing strategies and the standard expert FFN variants. P0/P1 items extend toward distributed multi-device dispatch, BASE / Megablocks-style efficiency, and end-to-end differentiable routing.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX kernels generated for all 7 entry points on `sm_75`
- [ ] Warp-level top-k reduction verified on Turing hardware

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels generated for `sm_80`, `sm_86`
- [ ] `cp.async`-staged expert weights for very large expert banks
- [ ] Tensor Core path for expert FFN GEMMs (16x16x16 / 16x8x16 tiles)

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [x] PTX kernels generated for `sm_90`, `sm_100`, `sm_120`
- [ ] TMA-based all-to-all dispatch primitive for distributed MoE
- [ ] `wgmma`-based grouped GEMM per expert for block-sparse dispatch (Megablocks pattern)
- [ ] Distributed shared-memory cluster reduction for load-balance loss across CTAs

---

## Deepening Opportunities

> Items marked `[x]` in the Completed section represent API and CPU-simulation coverage. The opportunities below close gaps toward production MoE serving and training.

### Verification Gaps
- [x] Top-k softmax normalisation: scores sum to 1.0 within `1e-4` for any input
- [x] Top-k indices in valid `[0, n_experts)` range
- [x] Switch capacity respected: no expert exceeds `ceil(T / E * cap_factor)` tokens
- [x] Switch overflow correctly counted when capacity < demand
- [x] Soft MoE dispatch weights row-sum to 1.0 within `1e-4`
- [x] PTX entry points validated for `.version`, `.visible .entry`, kernel name, and SM target across all 6 SM versions
- [ ] End-to-end Switch Transformer perplexity reproduction (paper baselines)
- [ ] Top-k GPU kernel correctness vs CPU simulation on `sm_80+`
- [ ] Distributed all-to-all primitive correctness on multi-GPU NCCL-equivalent

### Implementation Deepening
- [x] ExpertFfn output preserves input shape (`d_model` -> `d_model`) for all activations
- [x] SwiGLU output is finite for arbitrary input (no NaN propagation)
- [x] Load-balance loss is non-negative and finite for any logits / assignment combination
- [x] MoeLayer end-to-end forward produces shape-correct hidden states and finite auxiliary loss
- [ ] Megablocks-style block-sparse grouped GEMM (avoid padding to capacity)
- [x] BASE balanced assignment via Sinkhorn iterations
- [ ] Sparse upcycling: initialise MoE FFN weights from dense FFN checkpoint
- [ ] Multi-gate MoE for multi-task learning scenarios

## Notes

- Switch Transformer uses `drop_tokens = true` by default; tokens exceeding capacity are dropped and counted in `n_overflows`
- Expert Choice routing inverts the assignment direction: experts pick top-c tokens, guaranteeing perfectly balanced load at the cost of dropping low-score tokens
- Soft MoE replaces hard top-k with a fully differentiable per-slot softmax, eliminating overflow but increasing FLOPs
- `MoeLayerOutput` includes the auxiliary load-balance loss directly so the training loop can backprop without separate computation
- All PTX kernels share a unified `.version` / `.target sm_X` / `.address_size 64` header consistent with the rest of the OxiCUDA ecosystem
