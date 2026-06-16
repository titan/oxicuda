# oxicuda-anomaly TODO

Anomaly detection primitives for OxiCUDA (DeepSVDD, autoencoder / VAE reconstruction, LOF, k-NN, COPOD, Mahalanobis, Isolation Forest, MAD / Z-score, ensemble). Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.37).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

**Actual: ~23,650 SLoC (69 source files + 1 benches file) -- Coverage: deep / distance / density / statistical / ensemble anomaly families**

Current implementation covers all canonical anomaly-detection families: DeepSVDD (Ruff et al. 2018, 3-layer MLP with hypersphere-collapse prevention via no-bias last layer); autoencoder + VAE reconstruction-based scoring; LOF (Breunig et al. 2000) brute-force k-NN local outlier factor; pure k-NN distance baseline; COPOD (Li et al. 2020) empirical-CDF copula-based scoring with optional skewness adjustment; Mahalanobis distance with Gauss-Jordan covariance inversion and ridge stabilisation; Isolation Scorer (random-projection path-length estimation with `c(n) = 2H(n - 1) - 2(n - 1) / n` adjustment); MAD (`MAD = 1.4826 * median|x_i - mu|`) and Z-score (Welford online) statistical detectors; ensemble combiner (Average / Maximum / Weighted with per-detector min-max normalisation); AUC-ROC / AUC-PR / F1@threshold metrics.

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `AnomalyError` (variants for dimension mismatch, empty input, fit-required, invalid hyperparameters, NaN encountered, internal), `AnomalyResult<T>`
- [x] `handle.rs` -- `SmVersion(u32)`, `LcgRng` (Knuth MMIX 64-bit LCG with `next_normal` Box-Muller pair), `AnomalyHandle::default_handle()`

#### Deep Anomaly Detection
- [x] `svdd/deep_svdd.rs` -- DeepSVDD (Ruff et al. 2018): 3-layer MLP, no bias in last layer (hypersphere-collapse prevention); `fit` computes fixed center `c = mean(phi(x_i))` (adjusted if near zero); `score = ||phi(x) - c||^2`; Xavier init
- [x] `reconstruction/autoencoder.rs` -- `AeConfig`, `AutoencoderAnomaly` (Xavier init, ReLU encoder + Sigmoid decoder); `score = MSE` reconstruction error; batch scoring
- [x] `reconstruction/vae_anomaly.rs` -- `VaeConfig`, `VaeAnomaly` (mu / log_var encoder, Box-Muller reparametrize, MSE + KL beta-ELBO); deterministic-mu scoring (no stochasticity at inference)

#### Distance-Based Detection
- [x] `distance/lof.rs` -- LOF (Breunig et al. 2000): brute-force k-NN; `fit` computes `knn_indices` / `knn_dists` / `lrd`; `reach_dist_k(i, j) = max(knn_dists[j * (k - 1)], dist(i, j))`; `lrd_k(i) = k / sum(reach_dist)`; `score = mean(lrd_neighbors) / lrd_x`; numerical guard `lrd -> 1e30` if zero denominator
- [x] `distance/knn_score.rs` -- `KnnAnomalyScorer`: average k-NN distance baseline; brute-force; batch scoring
- [x] `distance/abod.rs` -- ABOD (Kriegel-Schubert-Zimek KDD 2008): ABOF=Var[⟨pa,pb⟩/(‖pa‖·‖pb‖)²]; score=-ABOF (high=anomalous); brute-force O(n²d); 14 unit tests
- [x] `distance/cblof.rs` -- CBLOF (He 2003): cluster-based LOF using k-means; score proportional to distance from nearest large cluster / intra-cluster distance; 12 unit tests
- [x] `distance/cof.rs` -- COF (Tang-Chen-Fu 2002): SBN greedy NN chain cost=(2/(k(k+1)))·Σᵢ i·dist(oᵢ₋₁,oᵢ); COF=cost/mean(neighbor costs); 17 unit tests

#### Density-Based Detection
- [x] `density/copod.rs` -- COPOD (Li et al. 2020): empirical CDF via sorted-column binary search; `score = -sum(log(F_j(x_j)) + log(1 - F_j(x_j))) / 2`; `score_skew_adjusted` (Fisher-Pearson skewness-weighted tail)
- [x] `density/mahalanobis.rs` -- `MahalanobisDetector`: sample mean + covariance estimation; Gauss-Jordan inversion with full pivoting on augmented `[M | I]`; ridge `0.01 * I` for numerical stability; `D^2 = diff^T * Sigma^(-1) * diff`
- [x] `density/fast_mcd.rs` -- FastMCD (Rousseeuw-Van Driessen JASA 1999): C-step MCD algorithm, n_starts random h-subsets, Cholesky log-det convergence, Gauss-Jordan inversion with ridge 1e-5; Mahalanobis² score w.r.t. robust covariance; 16 unit tests

#### Isolation
- [x] `isolation/iforest_score.rs` -- `IsolationScorer`: random-projection path-length estimation; `c_factor(n) = 2 * H(n - 1) - 2 * (n - 1) / n` (EULER_MASCHERONI = 0.5772156649); `isolation_score_from_path(avg_path, n) = 2^(-avg_path / c_n)`

#### Statistical Detection
- [x] `statistical/stats.rs` -- `MadDetector` (`MAD = 1.4826 * median|x_i - mu|`, Z-score via MAD, configurable threshold), `ZScoreDetector` (Welford online mu / sigma, `|x - mu| / sigma > threshold`), `percentile_threshold` (linear interpolation)

#### Ensemble
- [x] `ensemble/ensemble.rs` -- `AnomalyEnsemble` (`EnsembleMethod::Average` / `Maximum` / `Weighted` combiners with per-detector min-max normalisation); `add_detector`, `score_ensemble`

#### Metrics
- [x] `metrics/anomaly_metrics.rs` -- `auc_roc_anomaly`, `auc_pr` (precision-recall trapezoidal), `f1_at_threshold`, `compute_detection_metrics` (AUC-ROC + AUC-PR + F1 combined into `AnomalyDetectionMetrics`)

#### PTX Kernels
- [x] `ptx_kernels.rs` -- 7 GPU kernels x 6 SM versions (75/80/86/90/100/120):
  - [x] `svdd_loss_kernel` -- `||z - c||^2` per sample
  - [x] `recon_score_kernel` -- MSE reconstruction error per sample via warp-shuffle reduction
  - [x] `lof_reach_dist_kernel` -- k-distance lookup + max for LOF reachability
  - [x] `copod_ecdf_kernel` -- empirical-CDF binary-search rank
  - [x] `mahal_dist_kernel` -- quadratic form `diff^T * Sigma^(-1) * diff` per sample
  - [x] `iforest_score_kernel` -- `2^(-avg_path / c_n)` via `ex2.approx`
  - [x] `ensemble_normalize_kernel` -- per-detector min-max then mean / max / weighted combination

#### Integration Tests
- [x] 12 e2e tests (lib.rs): AE finite for train and noise, AE finite for arbitrary input, VAE finite, DeepSVDD score increases for far-away outlier, LOF finite for trivial uniform data, COPOD higher for extreme outlier, Mahalanobis higher for OOD point, Isolation score in `[0, 1]`, Z-score flags outlier, MAD finite, ensemble combine in `[0, 1]`, PTX kernels x 6 SM versions

#### Benchmarks
- [x] `benches/anomaly_ops.rs` -- 7 PTX kernel groups x 4 SM versions plus 5 algorithm benches

### Future Enhancements

#### P0 -- Critical Algorithmic Coverage
- [x] Trainable backward passes for DeepSVDD / autoencoder / VAE (`svdd/trainable_svdd.rs`)
- [x] Soft-boundary DeepSVDD (R^2 + nu-SVDD trade-off) in addition to one-class variant (`svdd/soft_svdd.rs`)
- [x] HBOS (Histogram-Based Outlier Score) -- per-feature histogram score, very fast baseline (`statistical/hbos.rs`)
- [x] PCA-based reconstruction anomaly score (subspace projection error) (`reconstruction/pca_anomaly.rs`)

#### P1 -- Important Features
- [x] Iforest tree construction kernel -- explicit tree splits (`isolation/iforest_tree.rs`)
- [x] Extended Isolation Forest (Hariri et al. 2018) -- random hyperplane splits instead of axis-aligned (`ensemble/ext_iforest.rs`)
- [x] ECOD (Empirical Cumulative Outlier Detection, Li et al. 2022) -- COPOD successor with skewness-aware tail (`statistical/ecod.rs`)
- [x] LODA (Lightweight On-line Detector of Anomalies, Pevny 2016) -- random-projection histograms (`ensemble/loda.rs`)
- [x] SUOD ensemble acceleration framework (subsampling + projection) (`ensemble/suod.rs`)
- [x] ROCK / IDEC tracker for streaming anomaly detection (`statistical/rock_idec.rs`)
- [x] GMM / kernel density estimator (KDE) detectors (`density/gmm_detector.rs`)
- [x] Approximate nearest-neighbour acceleration for LOF -- k-d tree CPU path (`distance/lof_kdtree.rs`)

#### P2 -- Advanced / Research
- [x] DAGMM (Deep Autoencoding Gaussian Mixture Model) (`reconstruction/dagmm.rs`)
- [x] AnoGAN / fAnoGAN GAN-based anomaly scoring (`reconstruction/anogan.rs`)
- [x] Diffusion-based anomaly detection -- tabular DDPM score network (`reconstruction/diffusion_anomaly.rs`)
- [x] Self-supervised anomaly detection (rotation prediction, jigsaw, contrastive) (`reconstruction/self_supervised.rs`)
- [x] Memory-augmented autoencoders (MemAE) (`reconstruction/mem_ae.rs`)
- [x] Conformal anomaly detection wrappers (distribution-free p-values) (`statistical/conformal.rs`)
- [x] Concept-drift-aware detectors for streaming (ADWIN, CUSUM, PHT, DDM) (`statistical/concept_drift.rs`)
- [x] Federated anomaly detection (`ensemble/federated.rs`)
- [x] Time-series anomaly detection (LSTM-AE, USAD, TranAD) — RNN-AE (`time_series/lstm_ae.rs`)
- [x] Graph anomaly detection (DOMINANT, AnomalyDAE) (`graph/dominant.rs`)
- [x] `isolation/inne.rs` — INNE (Isolation Nearest-Neighbour Ensemble, Bandaragoda 2018): isolation probability iz(x)=d(x,nn₁)/max{d(x,nnₖ)} via t-ball ratio per sample; ensemble average over ψ random sub-samples; O(ψ log n) per query
- [x] `svdd/deep_sad.rs` — DeepSAD (Ruff 2020, semi-supervised): hypersphere loss with labeled normal pulls (η=+1) + anomaly pushes (η=-1); η-weighted cross-entropy on labeled subset + DeepSVDD on unlabeled
- [x] `distance/lof_online.rs` — Online LOF (Pokrajac 2007): incremental O(k²) updates to kNN graph, LRD, and LOF scores on point insertion; avoids O(n²) full refit; supports streaming windows
- [x] `reconstruction/norm_flow.rs` — Normalising Flow anomaly scoring (Rezende-Mohamed 2015 extended): log-likelihood under RealNVP/MADE flow as anomaly score; OOD = low log p(x); uses existing `variational/real_nvp.rs` as reference
- [x] `statistical/extreme_value.rs` — Extreme Value Theory detector (Gnedenko 1943, Clifton 2011): GPD tail fitting on high-score exceedances via maximum-likelihood; automatic threshold selection by mean-excess plot; `GpdDetector`
- [x] `distance/abod_approx.rs` — FastABOD / approximate ABOD (Kriegel 2008 §4): restrict angle variance computation to k-NN set instead of all pairs; reduces O(n²d) to O(knd); `AbodApprox { k: usize }`
- [x] `distance/sod.rs` — SOD (Subspace Outlier Degree, Kriegel 2009): per-point shared-nearest-neighbour subspace projection; SOD=d(x,μ_snn)/Var_snn_subspace; handles high-dimensional feature irrelevance
- [ ] `ensemble/lscp.rs` — LSCP (Zhao 2019, Locally Selective Combination in Parallel): local pseudo-ground-truth via max-score in neighbourhood + greedy detector selection; `LscpEnsemble`
- [ ] Fused kNN+LOF PTX kernel: warp-level distance min reduction + concurrent reach-dist update in shared memory on sm_80+
- [ ] ABOD batch PTX kernel: stream-k angle-variance accumulation over query batches; 3-vector inner-product with reciprocal-distance weighting
- [ ] FastMCD GPU C-step: device-side Mahalanobis distance for all n points in one kernel call (replaces host-side loop in `density/fast_mcd.rs`)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| (none) | Standalone primitives crate | Yes |
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

## Quality Status

- Tests: 582 passing (12 e2e in lib.rs + module unit tests)
- All production code uses `Result` / `Option` (no `unwrap()` outside tests)
- `clippy::all` warnings: 0
- `missing_docs` warnings: 0
- Files: 69 source `.rs` files, all under 2000 lines
- GPU tests behind `#[cfg(feature = "gpu-tests")]`
- macOS compiles but returns `UnsupportedPlatform` at runtime

## Performance Targets

Representative shapes for canonical anomaly-detection benchmarks (ODDS, ADBench).

| Operation | Configuration | Priority |
|-----------|---------------|----------|
| `svdd_loss_kernel` | batch 1024, latent_dim 64 | P0 |
| `recon_score_kernel` | batch 1024, input_dim 128 | P0 |
| `lof_reach_dist_kernel` | 1000 samples, k = 20, d = 16 | P0 |
| `copod_ecdf_kernel` | 10K samples, d = 32 | P0 |
| `mahal_dist_kernel` | 1000 samples, d = 16 | P0 |
| `iforest_score_kernel` | 100 trees, 256 path entries | P1 |
| `ensemble_normalize_kernel` | 3-5 detectors, 1000 samples | P1 |

Target: scoring throughput comparable to PyOD CPU reference and (for deep detectors) PyTorch CUDA reference on `sm_80+`.

## Estimation vs Actual

| Metric | Description | Actual |
|--------|-------------|--------|
| Files | source `.rs` files under `src/` | 69 |
| SLoC | code lines (tokei) | ~23,650 |
| Tests | e2e + unit | 582 |
| Coverage | detector families | 5 (deep, distance, density, isolation, statistical) |
| Coverage | ensemble methods | 3 (Average, Maximum, Weighted) |

The current implementation provides a compact reference covering the five canonical anomaly-detection families used in the literature, plus three ensemble combination strategies and three calibrated metrics. P0/P1 items extend toward additional baseline detectors (HBOS, ECOD, LODA), trainable backward passes for the deep models, and explicit Isolation Forest tree construction.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX kernels generated for all 7 entry points on `sm_75`
- [ ] Warp-shuffled MSE reduction for `recon_score_kernel` verified on Turing

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX kernels generated for `sm_80`, `sm_86`
- [ ] `cp.async`-staged k-NN reference set for large LOF datasets
- [ ] Tensor Core path for Mahalanobis quadratic form on `d` multiple of 16
- [ ] Bank-conflict-free covariance-inverse staging in shared memory

### Hopper (sm_90) / Blackwell (sm_100, sm_120)
- [x] PTX kernels generated for `sm_90`, `sm_100`, `sm_120`
- [ ] TMA-based reference-set staging for very large k-NN reference sets
- [ ] `wgmma`-based batched Mahalanobis for high-dimensional Gaussian assumptions
- [ ] Distributed shared-memory cluster reduction for ensemble normalisation across many detectors

---

## Deepening Opportunities

> Items marked `[x]` in the Completed section represent API and CPU-simulation coverage. The opportunities below close gaps toward production anomaly-detection deployment.

### Verification Gaps
- [x] DeepSVDD: far-away point scores higher than centre point after fit (verified)
- [x] LOF: finite positive score for trivial uniform data (no NaN / Inf propagation)
- [x] COPOD: extreme outlier scores strictly higher than typical point
- [x] Mahalanobis: extreme outlier scores strictly higher than in-distribution point
- [x] IsolationScorer: score lies in `[0, 1]` for arbitrary input
- [x] Z-score / MAD: extreme outlier flagged as anomaly above threshold
- [x] Ensemble combine output is finite and lies in `[0, 1]` after min-max normalisation
- [x] PTX entry points validated for `.version`, `.visible .entry`, kernel name, and SM target across all 6 SM versions
- [ ] End-to-end ODDS / ADBench AUC-ROC reproduction against PyOD reference
- [ ] DeepSVDD GPU kernel correctness vs CPU simulation on `sm_80+`

### Implementation Deepening
- [x] Autoencoder produces finite reconstruction-MSE score for both in-distribution and noise inputs
- [x] VAE deterministic-mu score is finite for arbitrary input
- [x] LOF reachability distance and local reachability density correctly handle zero-distance neighbours via `1e30` guard
- [x] COPOD empirical CDF via sorted-column binary search supports arbitrary sample sizes
- [x] Mahalanobis Gauss-Jordan inversion with `0.01 * I` ridge handles near-singular covariance matrices
- [x] Trainable backward passes for DeepSVDD (`svdd/trainable_svdd.rs`)
- [x] Explicit Isolation Forest tree construction (`isolation/iforest_tree.rs`)
- [x] Approximate nearest-neighbour acceleration for LOF -- k-d tree CPU path (`distance/lof_kdtree.rs`)
- [x] Streaming / online updates for Z-score and MAD detectors (`statistical/online_stats.rs`)

## Notes

- DeepSVDD prevents hypersphere collapse by removing the bias of the final layer and using a fixed (not learnable) centre `c` computed from training data after the forward pass
- LOF reach-distance uses `max(k_th_neighbour_distance(j), dist(i, j))` to symmetrise the local-reachability density estimate
- COPOD scores both lower and upper tails using `-log(F_j(x_j)) - log(1 - F_j(x_j))` averaged across features, capturing both extreme small and extreme large values
- Mahalanobis estimation falls back to a `0.01 * I` ridge during Gauss-Jordan inversion to remain numerically stable for near-singular sample covariance matrices
- The ensemble combiner normalises each detector's score to `[0, 1]` via min-max from training scores before averaging (or taking max / weighted sum)
- All PTX kernels share a unified `.version` / `.target sm_X` / `.address_size 64` header consistent with the rest of the OxiCUDA ecosystem
