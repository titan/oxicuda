# oxicuda-stats TODO

GPU-accelerated statistical inference, hypothesis testing, and frequentist analysis,
serving as a pure Rust equivalent to SciPy's `stats` module + RAPIDS cuML's metrics.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.54).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 6,360 (71 files, including 5,342 code + 164 comments + 489 blanks; markdown 365)
- **Tests:** 160 passing (lib + e2e_tests)
- **Pure Rust:** Zero external linear-algebra dependencies; only `thiserror` runtime dep
- **PTX coverage:** 7 kernels x 6 SM versions = 42 PTX string generators

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `StatsError` enum (ShapeMismatch, NotConverged, EmptyInput, InvalidParameter, NumericalInstability, UnsupportedSmVersion, InsufficientSampleSize, DegreesOfFreedomZero, ProbabilityOutOfRange, SingularMatrix, IndexOutOfBounds, ...) + `StatsResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `StatsHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `mean_var`, `rank_assign`, `histogram_bin`, `bootstrap_resample`, `permute_labels`, `chi2_cell`, `lr_normal_eq` (string concatenation only, no nvcc dependency)

#### Special Functions
- [x] `special/erf.rs` -- Abramowitz-Stegun 7.1.26 series (validated erf(0) = 0, erf(1) ~ 0.8427, erf(2) ~ 0.9953)
- [x] `special/gammaln.rs` -- Lanczos approximation (validated lgamma(5) = ln(24), lgamma(0.5) = ln(sqrt(pi)))
- [x] `special/betainc.rs` -- Regularised incomplete beta via continued fraction (NR 6.4)
- [x] `special/lgamma_series.rs` -- Regularised lower incomplete gamma (gammp)
- [x] `special/digamma.rs` -- Asymptotic digamma series

#### Distributions
- [x] `distributions/normal.rs` -- PDF/CDF/PPF; ppf via Beasley-Springer-Moro
- [x] `distributions/student_t.rs` -- PDF/CDF/PPF; cdf at (t = 0, nu = 10) is exactly 0.5
- [x] `distributions/chi_squared.rs` -- PDF/CDF/PPF via regularised gamma
- [x] `distributions/f_dist.rs` -- PDF/CDF/PPF via regularised beta
- [x] `distributions/beta.rs` -- PDF/CDF/PPF via regularised beta
- [x] `distributions/gamma.rs` -- PDF/CDF/PPF via regularised gamma
- [x] `distributions/binomial.rs` -- PMF/CDF/PPF
- [x] `distributions/poisson.rs` -- PMF/CDF/PPF
- [x] `distributions/exponential.rs` -- PDF/CDF/PPF

#### Descriptive Statistics
- [x] `descriptive/summary.rs` -- mean / var / stddev / skewness / kurtosis
- [x] `descriptive/robust.rs` -- median, MAD, IQR, trimmed mean
- [x] `descriptive/quantile.rs` -- Empirical quantile types 1-9 (R / Hyndman-Fan)

#### Parametric Tests
- [x] `parametric/t_test.rs` -- One-sample Student, two-sample Student, Welch (Satterthwaite df), paired t
- [x] `parametric/anova.rs` -- One-way ANOVA ({1, 2, 3}, {3, 4, 5}, {5, 6, 7}) -> F = 12.0 matches SciPy; two-way row / col / interaction SS
- [x] `parametric/manova.rs` -- MANOVA Wilks lambda + Pillai trace + Hotelling-Lawley
- [x] `parametric/regression_inference.rs` -- Regression SE / t / F / R^2 / adj-R^2 / AIC / BIC

#### Nonparametric Tests
- [x] `nonparametric/mann_whitney.rs` -- Rank-based with tied-rank averaging + normal approximation
- [x] `nonparametric/wilcoxon.rs` -- Signed-rank exact for n < 25; normal approximation for n >= 25
- [x] `nonparametric/kruskal_wallis.rs` -- Rank H statistic with chi^2(k - 1) approximation
- [x] `nonparametric/friedman.rs` -- Within-block ranks with chi^2(k - 1) approximation

#### Goodness-of-Fit
- [x] `goodness_of_fit/ks.rs` -- KS one + two-sample with asymptotic Kolmogorov distribution
- [x] `goodness_of_fit/anderson_darling.rs` -- Anderson-Darling A^2 statistic
- [x] `goodness_of_fit/shapiro_wilk.rs` -- Shapiro-Wilk W (Royston coefficients)
- [x] `goodness_of_fit/jarque_bera.rs` -- Jarque-Bera chi^2(2) statistic

#### Contingency
- [x] `chi_squared/chi2_independence.rs` -- r x c independence test (Pearson chi^2)
- [x] `chi_squared/fisher_exact.rs` -- Hypergeometric Fisher exact (one + two-sided)
- [x] `chi_squared/mcnemar.rs` -- McNemar with continuity correction

#### Multiple Testing Correction
- [x] `multiple/bonferroni.rs` -- alpha / m
- [x] `multiple/holm.rs` -- Step-down Holm
- [x] `multiple/bh_fdr.rs` -- Benjamini-Hochberg FDR
- [x] `multiple/by_fdr.rs` -- Benjamini-Yekutieli FDR
- [x] `multiple/tukey_hsd.rs` -- Tukey HSD via studentized-range approximation

#### Resampling
- [x] `resampling/bootstrap.rs` -- B bootstrap replicates with statistic callback
- [x] `resampling/jackknife.rs` -- Leave-one-out jackknife variance
- [x] `resampling/permutation.rs` -- Permutation test for group label shuffling

#### Confidence Intervals
- [x] `ci/normal_ci.rs` -- Normal-z CI
- [x] `ci/t_ci.rs` -- Student-t CI
- [x] `ci/bootstrap_ci.rs` -- Bootstrap percentile + BCa
- [x] `ci/proportion_ci.rs` -- Wilson + Clopper-Pearson + Agresti-Coull

#### Regression
- [x] `regression/linear.rs` -- OLS via Cholesky on normal equations
- [x] `regression/logistic.rs` -- Logistic regression via IRLS
- [x] `regression/ridge_lr.rs` -- Ridge regression with lambda regularisation

#### Power Analysis
- [x] `power/t_power.rs` -- Sample size from (d, alpha, beta)
- [x] `power/anova_power.rs` -- Sample size for ANOVA
- [x] `power/effect_size.rs` -- eta^2, partial eta^2, omega^2

#### Correlation
- [x] `correlation/pearson.rs` -- Pearson r with t-test (df = n - 2)
- [x] `correlation/spearman.rs` -- Spearman via rank Pearson
- [x] `correlation/kendall_tau.rs` -- Kendall tau via concordant / discordant pairs

#### Validation
- [x] `e2e_tests.rs` -- 18 cross-module tests: erf / lgamma boundary values, Student-t self-symmetry, ANOVA F = 12.0, KS-1 normal small-D, Mann-Whitney identical groups -> U = n * m / 2, bootstrap CI contains true mean, OLS + inference returns expected SE / t / R^2, logistic IRLS classifies separable data, Wilks MANOVA two-group, Friedman ranks, Kendall tau on monotone, Wilson CI bracket coverage, Tukey HSD ordering, PTX x 6 SM
- [x] `benches/stats_ops.rs` -- Criterion: 7 PTX kernels x all SM + erf / t-test / KS / OLS / bootstrap algo benches

### Future Enhancements

#### P0 -- Critical
- [x] Generalised Linear Models (GLM) with link functions (Gaussian / Poisson / Binomial / Gamma / Inverse Gaussian) (`regression/glm.rs`)
- [x] Robust regression: Huber-M, RLM, RANSAC for outlier-resistant fits (`regression/robust.rs`)
- [x] Bayesian inference primitives: conjugate updates, credible intervals, Bayes factors (`bayesian/conjugate.rs`)

#### P1 -- Important
- [x] Time-series tests: Ljung-Box, Augmented Dickey-Fuller, KPSS, Box-Pierce (`time_series.rs`)
- [x] Mixed-effects models (LMM / GLMM) via EM + diagonal Woodbury (`regression/mixed_effects.rs`)
- [x] Quantile regression (Koenker-Bassett) via primal-dual IPM (`regression/quantile.rs`)
- [x] Negative binomial regression with iteratively reweighted least squares (`regression/negbinom.rs`)
- [x] Ordinal logistic / multinomial logistic with parallel-regression assumption tests (`regression/multinomial.rs`)
- [ ] Cox proportional hazards (delegate to `oxicuda-survival`; cross-link tests here)
- [x] Survey-design corrections: stratified / clustered / weighted variance estimators (`survey/design.rs`)

#### P2 -- Nice-to-Have
- [x] Permutation MANOVA (PERMANOVA) for high-dimensional data (`nonparametric/permanova.rs`)
- [x] Spatial statistics: Moran's I, Geary's C, Ripley's K (`spatial/spatial.rs`)
- [x] Circular statistics: von Mises distribution, Rayleigh test (`circular/circular.rs`)
- [x] Survival-free distributional tests: Cramer-von Mises, Watson (`goodness_of_fit/cvm_watson.rs`)
- [x] Multilevel / hierarchical bootstrap (cluster-resampling) (`resampling/multilevel_bootstrap.rs`)
- [x] Outlier-detection tests: Grubbs and Dixon Q (`nonparametric/outlier.rs`)
- [x] Empirical likelihood and exponential tilting for distribution-free inference (`resampling/empirical_likelihood.rs`)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean)
- Tests: 160 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- Pure Rust: no C/C++/Fortran in default features

## Performance Targets

Representative algorithmic benchmarks (CPU-side reference + PTX generation timing):

| Routine | Problem size | Priority |
|---------|--------------|----------|
| `erf` / `gammaln` / `betainc` | scalar | High |
| One-way ANOVA | k = 5, n_k = 100 | High |
| Welch t-test | n_1 = n_2 in {100, 1000} | High |
| KS one-sample | n in {1000, 10000} | High |
| OLS via Cholesky | (n, p) in {(1000, 10), (10000, 50)} | High |
| Bootstrap mean | n = 1000, B in {1000, 10000} | High |
| Mann-Whitney | n_1 = n_2 in {100, 1000} | Mid |
| Permutation test | n = 100, P in {1000, 5000} | Mid |

Target for GPU execution path: match SciPy / R `stats` numerical agreement within
4 significant digits and outperform CPU SciPy at n >= 10000 once `oxicuda-launch`
orchestrates the emitted PTX on Linux + NVIDIA.

## Notes

- All distribution PDFs / CDFs / PPFs evaluated to standard-library precision (f64).
- Newton-Raphson PPF solvers seed from inverse-CDF approximations (Beasley-Springer-Moro for normal, asymptotic for chi^2 / t).
- `LcgRng::box_muller` uses the polar-form Box-Muller (no rejection) for normal sampling.
- All p-values clipped to `[0.0, 1.0]`; one-tail vs two-tail toggles via `Tail` enum.
- Multiple-testing routines accept either raw p-values or test statistics with degrees of freedom.

---

## Architecture-Specific Deepening

### PTX Coverage Matrix

| Kernel | sm_70 | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|--------|-------|-------|-------|-------|-------|-------|
| `mean_var` | [x] | [x] | [x] | [x] | [x] | [x] |
| `rank_assign` | [x] | [x] | [x] | [x] | [x] | [x] |
| `histogram_bin` | [x] | [x] | [x] | [x] | [x] | [x] |
| `bootstrap_resample` | [x] | [x] | [x] | [x] | [x] | [x] |
| `permute_labels` | [x] | [x] | [x] | [x] | [x] | [x] |
| `chi2_cell` | [x] | [x] | [x] | [x] | [x] | [x] |
| `lr_normal_eq` | [x] | [x] | [x] | [x] | [x] | [x] |

All six SM versions produce non-empty PTX strings and pass content-substring checks in `e2e_tests.rs`.

### Per-Architecture Optimisation Hooks
- [ ] sm_80 (Ampere) -- warp-shuffle reductions for `mean_var` and `chi2_cell`
- [ ] sm_89 (Ada) -- `cp.async` for `histogram_bin` global-to-shared bucket counters
- [ ] sm_90 (Hopper) -- TMA prefetch for `bootstrap_resample` input tiles
- [ ] Verify `rank_assign` produces stable ranks under ties on all SM versions

---

## Deepening Opportunities

### Verification Gaps (require Linux + NVIDIA hardware)
- [ ] GPU run of all 7 PTX kernels under `cargo nextest --features gpu-tests` on sm_80 / sm_89 / sm_90
- [ ] Numerical agreement between CPU reference and GPU `mean_var` within 1 ULP for f32 and f64
- [ ] Bootstrap throughput at B = 10000 vs CPU SciPy on n = 10000

### Algorithmic Deepening
- [x] Higher-order moment estimators (L-moments, probability-weighted moments) (`descriptive/lmoments.rs`)
- [x] Stable estimation under heavy tails via trimmed / winsorised statistics (`descriptive/lmoments.rs`)
- [x] Optimised Fisher exact for large contingency tables (logarithmic hypergeometric) (`chi_squared/fisher_exact_fast.rs`)
- [x] Vectorised permutation test that re-uses the same shuffle across many statistics (`resampling/vectorised_permutation.rs`)
- [x] Sequential testing with early stopping (group-sequential / alpha-spending) (`multiple/sequential.rs`)

### API Polish
- [x] Builder-pattern `TTestBuilder`, `AnovaBuilder`, `BootstrapBuilder` configurators (`parametric/test_builder.rs`)
- [ ] Convenience traits to consume `ndarray::Array1` / `ndarray::Array2` once
  the optional `ndarray` feature lands
- [ ] Cross-link with `oxicuda-survival` for hazard regression and with `oxicuda-cvx`
  for constrained MLE (e.g., box-constrained quantile regression)
