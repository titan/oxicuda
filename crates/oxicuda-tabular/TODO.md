# oxicuda-tabular TODO

Tabular deep-learning primitives for OxiCUDA (sparsemax / entmax-1.5, TabNet, SAINT, FT-Transformer, NODE, normalizers, classification metrics). Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.36).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: 23,054 SLoC (49 source files + 1 benches file) -- Coverage: sparse-attention probability transforms + 4 canonical tabular DL models + preprocessing + metrics**

Current implementation covers the canonical tabular deep-learning toolkit: sparsemax (Martins & Astudillo 2016 sort-descending k* algorithm, O(d log d)) and entmax-1.5 (alpha = 1.5 via 64-iteration bisection); TabNet (Arik & Pfister 2021) with GLU gates, BatchNorm1d, step-wise sparsemax attention, prior scales `P_i = product(gamma - M_j)` and shared + step-specific FC-BN-GLU blocks; SAINT (Somepalli et al. 2021) with row-wise multi-head self-attention plus inter-sample attention, Pre-LayerNorm FFN, and CLS mean-pool head; FT-Transformer (Gorishniy et al. 2021) with continuous feature tokenisation `x_j * w_j + b_j` per embedding dimension, categorical lookup tables, Pre-LN MHSA blocks, CLS token and linear head; NODE (Popov et al. 2019) soft oblivious decision trees with entmax-1.5 feature selection and sigmoid-smoothed splits, plus `NodeEnsemble` mean-over-trees; quantile / standard / min-max normalisers; classification metrics including AUC-ROC trapezoidal integration.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `TabularError`, `TabularResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng`, `TabularHandle::default_handle()`

#### Sparse Probability Transforms
- [x] `attention/sparsemax.rs` -- `sparsemax` (Martins & Astudillo 2016 sort-descending k* algorithm, O(d log d)), `entmax15` (alpha = 1.5 via 64-iteration bisection), `sparsemax_batch` for batched row-wise simplex projection

#### Tabular Networks
- [x] `attention/tabnet.rs` -- TabNet (Arik & Pfister 2021): `glu` (GLU gate), `BatchNorm1d` (learnable gamma / beta), `TabNetConfig { n_features, n_d, n_a, n_steps, gamma, n_classes }`, `TabNetLayer` (Xavier init, step-wise sparsemax attention, prior scales `P_i = product(gamma - M_j)`, shared + step-specific FC-BN-GLU blocks)
- [x] `attention/saint.rs` -- SAINT (Somepalli et al. 2021): `self_attention` (scaled dot-product), `multihead_attention`, `intersample_attention` (per-feature cross-sample), `SaintConfig`, `SaintLayer` (alternating row-wise plus inter-sample MHSA with Pre-LN FFN, CLS mean-pool head)
- [x] `transformer/ft_transformer.rs` -- FT-Transformer (Gorishniy et al. 2021): `FeatureTokenizer` (continuous: `x_j * w_j + b_j` per embed dim; categorical: lookup table), `FtConfig`, `FtTransformer` (Pre-LN MHSA blocks, CLS token, linear head)
- [x] `tree/node.rs` -- NODE (Popov et al. 2019): `NodeConfig`, `NodeTree` (depth-d soft oblivious decision tree, entmax-1.5 feature selection, sigmoid-smoothed splits, leaf tensor products), `NodeEnsemble` (mean over trees)

#### Preprocessing
- [x] `preprocess/normalize.rs` -- `QuantileNormalizer` (empirical rank `[0, 1]`, binary-search transform), `StandardNormalizer` (z-score, Welford-style standard deviation), `MinMaxNormalizer`
- [x] `preprocess/embed.rs` -- `FeatureEmbedder` (fit continuous `mu` / `sigma`, validate categorical ranges)

#### Metrics
- [x] `metrics/tabular_metrics.rs` -- `binary_accuracy`, `multiclass_accuracy` (argmax), `rmse`, `mae`, `auc_roc` (sort by score + trapezoidal rule), `compute_binary_metrics`, `ClassificationMetrics`

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 7 GPU kernels x 6 SM versions (75/80/86/90/100/120):
  - [x] `sparsemax_kernel` -- per-row simplex projection
  - [x] `feature_tokenize_kernel` -- FT-Transformer continuous feature tokenisation
  - [x] `tabnet_step_attn_kernel` -- prior-scaled `Q.K^T` dot product + sparsemax
  - [x] `intersample_attn_kernel` -- SAINT cross-sample `Q.K^T / sqrt(d)`
  - [x] `node_tree_eval_kernel` -- soft oblivious routing plus leaf weighting
  - [x] `quantile_norm_kernel` -- binary-search empirical-rank computation
  - [x] `auc_roc_kernel` -- sorted-label trapezoidal area accumulation

#### Integration Tests
- [x] 12 e2e tests (lib.rs): sparsemax sums to 1, sparsemax one-hot for dominated input, entmax15 sums to 1, GLU halves dimension, TabNet output shape, TabNet attention masks non-negative, FT-Transformer finite logits, FeatureTokenizer shape, NODE forward shape, QuantileNormalizer range `[0, 1]`, AUC-ROC perfect predictor = 1.0, PTX kernels x 6 SM versions

#### Benchmarks
- [x] `benches/tabular_ops.rs` -- 7 PTX kernel groups x 4 SM versions plus 5 algorithm benches

### Future Enhancements

#### P0 -- Critical Algorithmic Coverage
- [x] Trainable backward passes (explicit analytic gradients) for all four attention/tree tabular models, each verified by a central finite-difference gradient check:
  - [x] `transformer/ft_transformer_grad.rs` -- `FtTransformer::backward` (softmax-Jacobian MHSA, QKV/Wo projections, GELU FFN, both LayerNorms, CLS head, tokenizer); `FtGradients`; FD rel-tol `< 3.5e-2` (params) / `< 2e-2` (input) on a tiny model
  - [x] `attention/tabnet_grad.rs` -- `TabNetLayer::backward` (recurrent prior-scale chain `P_{i+1}=P_i⊙(γ−M_i)`, sparsemax support-Jacobian, BN γ/β, shared+step GLU/FC, mean-pool head); `TabNetGradients`; FD rel-tol `< 5e-2`
  - [x] `attention/saint_grad.rs` -- `SaintLayer::backward` (per-sample row MHSA + per-feature inter-sample MHSA via shared `mhsa_backward`, all 4 LayerNorms incl. post-LN on FFN output, GELU FFN, mean-pool head); `SaintGradients`; FD rel-tol `< 4e-2` (abs/rel combined for tiny-gradient entries)
  - [x] `tree/node_grad.rs` -- `NodeTree::backward` / `NodeEnsemble::backward` (entmax-1.5 Jacobian `∂p_i/∂z_j = 2 s_i(δ_ij − s_j/Σs)`, sigmoid-smoothed splits, leaf-product routing, ensemble mean); `NodeTreeGradients`; FD rel-tol `< 5e-2`
- [x] Joint categorical + continuous tabular pipeline with mixed-type encoders end-to-end (`transformer/unified_encoder.rs` -- `JointTokenizer` jointly embeds continuous rank-1 affine tokens `x_j·w_j+b_j` AND per-category lookup embeddings into one merged sequence; `UnifiedEncoder` prepends optional CLS + Pre-LN MHSA blocks; `tokenize` / `forward` / `pooled`; verified token-count/shape, independent continuous & categorical contributions, finite + deterministic forward, cont-only / cat-only configs)
- [x] Self-supervised pretraining objectives for tabular data (denoising / contrastive) (`preprocess/ssl_pretrain.rs` -- VIME (Yoon et al. 2020): mask-estimation BCE + feature-reconstruction MSE over empirical-marginal corruption; SCARF (Bahri et al. 2022): InfoNCE contrastive loss over feature-resampling corruption)
- [x] CutMix / Mixup augmentation primitives for tabular learning

#### P1 -- Important Features
- [x] AutoInt (Song et al. 2019) -- multi-head self-attentive feature interactions
- [x] TabTransformer (Huang et al. 2020) -- transformer over categorical embeddings + continuous concat
- [x] DCN v2 (Wang et al. 2021) -- deep + cross network with low-rank factorisation
- [x] DANets -- deep abstract network with abstract layers (danet.rs -- Chen 2022; abstract layer with sparsemax sparse feature-group masks + affine aggregation, stacked with shortcuts)
- [x] DeepGBM -- gradient-boosting + neural hybrid (deepgbm.rs -- Ke 2019; GBDT2NN leaf-index-embedding MLP + CatNN FM over categorical features, combined → CTR; forward-only with provided leaf assignments)
- [x] TabPFN-style transformer-based prior-fit network (small-dataset specialist) (`transformer/tabpfn.rs` -- Hollmann et al. 2023; in-context classifier: shared feature encoder + per-class label embeddings, support/query in one sequence, causal-style in-context attention mask (support never attends to queries), softmax head over query tokens; forward / prior-fit inference)
- [x] Calibration metrics (ECE, MCE, reliability diagrams) (`metrics/calibration.rs` -- Guo et al. 2017 temperature scaling + Naeini et al. 2015 reliability binning; ECE/MCE, equal-width/equal-mass bins, multi-class Brier, Newton-fit `TemperatureScaler`)
- [x] PR-AUC (precision-recall area under curve) and log-loss / Brier score metrics

#### P2 -- Advanced / Research
- [x] FT-Transformer with attention-bias / RoPE (`transformer/ft_rope.rs` -- RoPE rotary Q/K (Su et al. 2021 RoFormer) + T5-style learnable per-head relative attention-bias table (Raffel et al. 2020), Pre-LN blocks + CLS head)
- [x] NODE with TabRecord / VarOblivious variants
- [x] Diffusion models for tabular generation (TabDDPM)
- [x] GANs for tabular data (CTGAN / TVAE) generators (CTGAN in `gan/ctgan.rs` -- Xu 2019; mode-specific normalisation, conditional sampler, PacGAN discriminator. TVAE in `vae/tvae.rs` -- Xu 2019; mode-normalised VAE with KL + reconstruction ELBO)
- [x] Conformal prediction wrappers for distribution-free uncertainty (`conformal/split_conformal.rs` -- Vovk 2005 / Lei et al. 2018 split conformal, Romano et al. 2019 CQR, Romano et al. 2020 APS + Sadinle et al. 2019 LAC; finite-sample `(n+1)` empirical quantile)
- [x] Federated tabular learning primitives (split / vertical / horizontal) (`federated.rs` -- horizontal_split (row partition) + vertical_split (column partition); FedAvg sample-weighted aggregation (McMahan 2017); FedProx proximal penalty + gradient (Li 2020); `SecureAggregator` pairwise-cancelling additive masks (Bonawitz 2017, simplified))
- [x] Concept-drift detection for streaming tabular features
- [x] Differentiable feature selection / importance attribution (`feature_select/stg.rs` -- STG stochastic gates, Yamada et al. 2020; L0-surrogate regulariser + per-feature learned `importances()` + `selected_features` thresholding)
- [x] `transformer/node.rs` — NODE (Neural Oblivious Decision Ensembles, Popov 2019): differentiable oblivious trees with entmax-split threshold learning + ensemble averaging — ALREADY EXISTS as `tree/node_oblivious.rs` (`ObliviousTree`, `NodeObliviousLayer`, `entmax_alpha_f64` / `entmoid_alpha_f64`, `EnsembleReduction`)
- [x] `diffusion/tabddpm.rs` — TabDDPM (Kotelnikov 2023): Gaussian DDPM for continuous + multinomial diffusion for categorical features; denoising UNet with timestep embedding; generation + anomaly scoring via ELBO
- [x] `gan/ctgan.rs` — CTGAN (Xu 2019): conditional GAN with mode-specific normalisation for imbalanced categoricals; PacGAN discriminator packing; training-by-sampling from conditional distributions; `CtGan { pac: usize }` — IMPLEMENTED in `gan/ctgan.rs` (`CtGan`, `ConditionalSampler`, `ModeNormalizer`, `ColumnModes`, discriminator/generator steps)
- [x] `preprocess/target_encode.rs` — Target encoding with regularisation (Micci-Barreca 2001): replace categorical level c with E[y|x=c] smoothed by global prior; smoothing factor k; leave-one-out for train/test; `TargetEncoder { k: f32, min_count: usize }`
- [x] `preprocess/quantile_feat.rs` — Quantile feature transformation (scikit-learn QuantileTransformer): map each feature to empirical quantile → Gaussian or uniform output; `QuantileTransformer { n_quantiles: usize, output_dist: QuantileDist }`
- [x] `conformal/aps_conformal.rs` — APS (Adaptive Prediction Set, Romano 2020): conformity score for multi-class: include classes in decreasing probability order until cumulative mass ≥ α; `ApsConformal { alpha: f32 }` — complement to existing split conformal

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none) | Standalone primitives crate | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

## Quality Status

- Tests: 564 passing (14 e2e in lib.rs + module unit tests, incl. 21 new backward / unified-encoder gradient-check tests)
- All production code uses `Result` / `Option` (no `unwrap()` outside tests)
- `clippy::all` warnings: 0
- `missing_docs` warnings: 0
- Files: 58 source `.rs` files, all under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns `UnsupportedPlatform` at runtime

## Performance Targets

Representative shapes for canonical tabular benchmarks (Forest Cover Type, Higgs, Income, etc.).

| Operation | Configuration | Priority |
|-----------|---------------|----------|
| `sparsemax_kernel` | batch 256, d in {64, 128, 256, 512} | P0 |
| `feature_tokenize_kernel` | batch 256, n_feat 32, embed_dim 64 | P0 |
| `tabnet_step_attn_kernel` | batch 256, n_feat 32, n_steps 4 | P0 |
| `intersample_attn_kernel` | batch 256, n_feat 32, n_heads 4 | P0 |
| `node_tree_eval_kernel` | 100 trees, depth 6, n_feat 32 | P0 |
| `quantile_norm_kernel` | 100K samples, 32 features | P1 |
| `auc_roc_kernel` | 100K predictions | P1 |

Target: forward latency comparable to PyTorch + pytorch-tabular reference for FT-Transformer and TabNet on `sm_80+`.

## Estimation vs Actual

| Metric | Description | Actual |
|--------|-------------|--------|
| Files | source `.rs` files under `src/` | 58 |
| SLoC | code lines (tokei) | ~23,573 |
| Tests | e2e + unit | 564 |
| Coverage | tabular DL models | 4 (TabNet, SAINT, FT-Transformer, NODE) |
| Coverage | normalizers | 3 (Quantile, Standard, MinMax) |

The current implementation provides a compact reference covering the four most-cited tabular deep-learning architectures, both standard sparse-attention probability transforms, and the canonical preprocessing / evaluation utilities. P0/P1 items extend toward trainable backward passes, additional architectures (AutoInt, TabTransformer, DCNv2), and richer metrics / calibration.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX kernels generated for all 7 entry points on `sm_75`
- [ ] Warp-shuffled top-k for TabNet sparsemax verified on Turing (requires GPU hardware)

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels generated for `sm_80`, `sm_86`
- [ ] `cp.async`-staged feature embeddings for large categorical vocabularies (requires GPU hardware)
- [ ] Tensor Core path for SAINT / FT-Transformer MHSA (16x16x16 / 16x8x16 tiles) on hidden multiples of 16 (requires GPU hardware)
- [ ] Bank-conflict-free NODE leaf table lookup (requires GPU hardware)

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [x] PTX kernels generated for `sm_90`, `sm_100`, `sm_120`
- [ ] `wgmma`-based FT-Transformer block for large `embed_dim` (requires GPU hardware)
- [ ] TMA-based categorical lookup for very large vocab sizes (requires GPU hardware)
- [ ] Distributed shared-memory cluster reduction for AUC-ROC across large prediction batches (requires GPU hardware)

---

## Deepening Opportunities

> Items marked `[x]` in the Completed section represent API and CPU-simulation coverage. The opportunities below close gaps toward production tabular DL deployment.

### Verification Gaps
- [x] sparsemax output sums to 1.0 within `1e-5` for any input distribution
- [x] sparsemax produces one-hot output when dominated (test: 50, 0, 0, 0 -> 1, 0, 0, 0)
- [x] entmax15 output sums to 1.0 within `1e-2` (bisection tolerance)
- [x] AUC-ROC = 1.0 for perfect predictor (well-separated scores)
- [x] QuantileNormalizer output lies in `[0, 1]` for in-distribution data
- [x] PTX entry points validated for `.version`, `.address_size 64`, `.visible .entry`, kernel name, and SM target across all 6 SM versions
- [ ] End-to-end OpenML / UCI benchmark accuracy reproduction (TabNet / SAINT / FT-Transformer / NODE papers) (requires real datasets + full training infrastructure)
- [ ] TabNet GPU kernel correctness vs CPU simulation on `sm_80+` (requires GPU hardware)

### Implementation Deepening
- [x] TabNet attention masks are non-negative (verified by `e2e_tabnet_attention_valid`)
- [x] FT-Transformer produces finite logits with correct output shape for arbitrary mixed continuous + categorical input
- [x] NODE ensemble forward returns correct `output_dim` shape for any tree count + depth
- [x] FeatureTokenizer produces `(n_cont + n_cat) * embed_dim` tokens
- [x] Trainable backward passes for TabNet step-wise attention prior scales (`attention/tabnet_grad.rs` -- the recurrent `P_{i+1}=P_i⊙(γ−M_i)` chain is differentiated exactly; the prior gradient of a late step flows back into every earlier step's prior, FD-verified via `bn_gamma` / `bn_beta` / `att_w` checks)
- [ ] Mixed-precision (FP16 attention + FP32 master parameters) for memory-bound large vocabularies (requires GPU hardware)
- [x] Calibration metrics (ECE / MCE) and reliability diagrams (`metrics/calibration.rs` -- Guo et al. 2017 / Naeini et al. 2015; duplicate of the P1 calibration item, implemented once)
- [x] PR-AUC and log-loss / Brier score for class-imbalanced tabular problems (`metrics/pr_metrics.rs` -- `pr_auc`, `average_precision`, `log_loss` / `multiclass_log_loss`, `brier_score`, full `precision_recall_curve`)

## Notes

- Sparsemax is the building block for both TabNet step-wise attention and NODE soft oblivious feature selection
- TabNet `prior_scales[i] = product_{j < i}(gamma - M_j)` enforces sequential feature selection across attention steps
- SAINT alternates row-wise self-attention with inter-sample attention to enable cross-sample feature interactions
- FT-Transformer treats each continuous feature as a learnable token (rank-1 embedding `x_j * w_j + b_j`) and each categorical feature as a lookup-table token
- NODE soft trees use entmax-1.5 to select features at each depth and sigmoid-smoothed splits to obtain a differentiable approximation to oblivious decision trees
- All PTX kernels share a unified `.version` / `.target sm_X` / `.address_size 64` header consistent with the rest of the OxiCUDA ecosystem
