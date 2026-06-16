# oxicuda-survival TODO

GPU-accelerated survival analysis and time-to-event modelling,
serving as a pure Rust equivalent to R's `survival`, Python's `lifelines`,
and `scikit-survival`.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.56).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** grown well beyond the original 7,367; see `wc -l src/**/*.rs` (includes Cox residual/influence diagnostics + Aalen-Johansen variance modules)
- **Tests:** 785 passing (lib + e2e_tests) via `cargo nextest run -p oxicuda-survival --all-features`
- **Pure Rust:** Zero external linear-algebra dependencies; only `thiserror` runtime dep
- **PTX coverage:** 7 kernels x 6 SM versions = 42 PTX string generators

### Completed

#### Core Infrastructure
- [x] `error.rs` -- `SurvivalError` enum (14 variants: ShapeMismatch, NotConverged, EmptyDataset, NoEvents, InvalidParameter, NumericalInstability, UnsupportedSmVersion, NegativeTime, SingularMatrix, IndexOutOfBounds, DimensionMismatch, ...) + `SurvivalResult<T>`
- [x] `handle.rs` -- `SmVersion`, `LcgRng` (MMIX LCG, bit-32 bool, Box-Muller normal), `SurvivalHandle`
- [x] `ptx_kernels.rs` -- 7 kernels x 6 SM versions: `km_step`, `cox_risk_sum`, `cox_score`, `cox_info`, `logrank_oe`, `brier_score`, `rmst_integrate` (string concatenation only, no nvcc dependency)

#### Data Primitives
- [x] `data/observation.rs` -- `Observation { time, event }`
- [x] `data/dataset.rs` -- `Dataset` with optional covariates and strata
- [x] `data/risk_set.rs` -- Sorted-time risk-set builder

#### Nonparametric Estimators
- [x] `nonparametric/kaplan_meier.rs` -- KM `S(t) = product (1 - d_i / n_i)` + Greenwood `Var(log S) = sum d_i / (n_i (n_i - d_i))` + log-log pointwise CIs
- [x] `nonparametric/nelson_aalen.rs` -- Cumulative hazard `H(t) = sum d_i / n_i` with variance
- [x] `nonparametric/life_table.rs` -- Discrete-interval (actuarial) life table
- [x] `nonparametric/survival_function.rs` -- Survival-function utilities and curve sampling

#### Hypothesis Tests
- [x] `test/log_rank.rs` -- K-sample log-rank with hypergeometric variance, chi^2(k - 1)
- [x] `test/stratified_log_rank.rs` -- Stratified summing of O - E and V
- [x] `test/peto_peto.rs` -- Peto-Peto weights `w_t = S(t)`
- [x] `test/gehan_breslow.rs` -- Gehan-Breslow weights `w_t = n_t`

#### Cox Proportional Hazards
- [x] `cox/cox_ph.rs` -- Partial likelihood orchestrator
- [x] `cox/breslow_ties.rs` -- Breslow tie handling
- [x] `cox/efron_ties.rs` -- Efron tie handling
- [x] `cox/newton_raphson.rs` -- Newton-Raphson with line search
- [x] `cox/schoenfeld.rs` -- Schoenfeld residuals `r_i = x_i - x_bar_R`
- [x] `cox/baseline_hazard.rs` -- Breslow baseline `H_0(t) = sum d_i / sum exp(beta^T x_j)`
- [x] `cox/residuals_diagnostic.rs` -- Martingale residuals `M_i = delta_i - H_0(t_i) exp(beta^T x_i)` (sum to 0), deviance residuals `d_i = sign(M_i) sqrt(-2[M_i + delta_i ln(delta_i - M_i)])`, Lin-Wei-Ying cumulative martingale process + sup statistic
- [x] `cox/influence_diagnostics.rs` -- Efficient score residuals `L_i = integral (x_i - x_bar) dM_i`, DFBeta `= L_i I(beta)^-1`, standardised DFBetas, likelihood displacement `LD_i = DFBeta_i^T I DFBeta_i` (validated vs leave-one-out refit)

#### Accelerated Failure Time
- [x] `aft/exponential.rs` -- Closed-form MLE
- [x] `aft/weibull.rs` -- Newton on right-censored log-likelihood
- [x] `aft/log_normal.rs` -- Newton on right-censored log-likelihood
- [x] `aft/log_logistic.rs` -- Newton on right-censored log-likelihood
- [x] `aft/generalized_gamma.rs` -- Numerical-gradient ascent
- [x] `aft/fit_aft.rs` -- Unified AFT fit driver with model selection

#### Time-Varying Cox
- [x] `time_varying/time_varying_cox.rs` -- (start, stop, event, x(t)) intervals
- [x] `time_varying/counting_process.rs` -- Counting-process risk-set membership based on (start, stop)

#### Competing Risks
- [x] `competing/cumulative_incidence.rs` -- CIF `F_k(t) = sum S(t_i^-) * d_{k,i} / n_i`
- [x] `competing/cause_specific_hazard.rs` -- Cause-specific Cox per event type
- [x] `competing/fine_gray.rs` -- Sub-distribution hazard with IPCW weights `w(t) = G(t) / G(t_i)`

#### Restricted Mean Survival Time
- [x] `rmst/rmst_estimator.rs` -- `RMST(tau) = integral_0^tau S(t) dt` via rectangle integration
- [x] `rmst/restricted_mean.rs` -- Delta-method variance and two-arm comparisons

#### Concordance
- [x] `concordance/harrell_c.rs` -- Harrell C over comparable pairs
- [x] `concordance/uno_c.rs` -- Uno IPCW-weighted concordance

#### Calibration
- [x] `calibration/brier_score.rs` -- Naive Brier score at horizon
- [x] `calibration/ipcw_brier.rs` -- IPCW Brier with censoring correction
- [x] `calibration/integrated_brier.rs` -- Integrated Brier over [0, tau]
- [x] `calibration/time_dependent_auc.rs` -- Time-dependent AUC (cumulative incidence vs survivor)

#### Deep-Learning Bridge
- [x] `deep/deepsurv_head.rs` -- DeepSurv-style log-risk head
- [x] `deep/partial_likelihood_grad.rs` -- Gradient of Cox partial likelihood wrt log-risk eta
- [x] `deep/surv_loss.rs` -- `cox_loss`, `brier_loss` PyTorch-style callables

#### Private Linear Algebra
- [x] `linalg/matmul.rs` -- Dense matmul
- [x] `linalg/cholesky.rs` -- Cholesky factorisation for Fisher information
- [x] `linalg/solve.rs` -- Triangular solves
- [x] `linalg/inverse.rs` -- Gauss-Jordan inverse with determinant

#### Special Functions
- [x] `special/gammaln.rs` -- Lanczos `gammaln`
- [x] `special/digamma.rs` -- Asymptotic `digamma`; Acklam normal-inverse helper

#### Summaries
- [x] `metrics/metrics.rs` -- Median survival, restricted mean, S(t) at horizon

#### Validation
- [x] `e2e_tests.rs` -- 30 cross-module tests: KM exact recovery, Greenwood SE, Cox beta recovery within 5%, Newton-Raphson < 50 iterations, Schoenfeld sum = 0, log-rank permutation invariance, Harrell C = 1.0 perfectly ranked / ~ 0.5 random, RMST on constant S, Fine-Gray reduces to Cox without competing events, Weibull MLE on exponential recovers k ~ 1, PTX x 6 SM
- [x] `benches/survival_ops.rs` -- Criterion: 7 PTX kernels x all SM + KM / Cox-Newton / log-rank / RMST algo benches

### Future Enhancements

#### P0 -- Critical
- [x] Penalised Cox: ridge (L2), lasso (L1) via coordinate descent (`cox/penalized_cox.rs`)
- [x] Frailty models (gamma / log-normal frailty) for clustered survival data (`nonparametric/frailty.rs`)
- [x] Joint longitudinal-survival models (linked to `oxicuda-stats` GLMs) (`longitudinal/joint_model.rs`)

#### P1 -- Important
- [x] Aalen additive hazards model `lambda(t | x) = sum beta_j(t) x_j` (`nonparametric/aalen.rs`)
- [x] Pseudo-observations for restricted mean / cumulative incidence regression (`rmst/pseudo_obs.rs`)
- [x] Royston-Parmar flexible parametric models (spline log-cumulative-hazard) (`aft/royston_parmar.rs`)
- [x] Discrete-time survival via complementary log-log / logistic link (`aft/discrete_time.rs`)
- [x] Inverse-probability-of-treatment weighting for causal hazard estimation (`cox/iptw.rs`)
- [x] Multi-state models (Markov + semi-Markov transitions) via Aalen-Johansen estimator (`nonparametric/multi_state.rs`)
- [x] Aalen-Johansen variance & confidence bands: recursive Greenwood-type covariance of the product integral + log-transform pointwise CIs; competing-risks CIF with variance; exact reduction to KM Greenwood for the 2-state case (`nonparametric/multi_state_inference.rs`)
- [x] Truncation (left, right, interval) support in `Dataset` (`data/truncation.rs`)
- [x] Predictive performance: time-dependent ROC curves, calibration plots, decision curves (`calibration/time_roc.rs`)

#### P2 -- Nice-to-Have
- [x] Bayesian survival (MCMC) for Weibull / log-normal / Cox-Bayes (`bayes/mcmc_survival.rs`)
- [x] Survival random forests (Ishwaran et al. 2008) with log-rank splitting (`nonparametric/survival_rf.rs`)
- [x] Gradient-boosted Cox (XGBoost-style) on log-risk (`cox/gradient_boost.rs`)
- [x] Recurrent-event models: Andersen-Gill, marginal means / rates (`nonparametric/recurrent.rs`)
- [x] Cure models (mixture-cure with logistic susceptibility) (`cox/cure_model.rs`)
- [x] Survival meta-analysis: combining KM curves from multiple studies (`nonparametric/survival_meta.rs`)
- [x] Power / sample-size calculations for survival trials (Schoenfeld formula) (`test/power_sample_size.rs`)
- [x] Net survival / relative survival (cancer-registry methods) (`nonparametric/net_survival.rs`)

#### P3 — v0.2.0 Extension Targets

- [x] `cox/landmark.rs` — Landmarking (Van Houwelingen 2007): dynamic landmark supermodels; fit Cox at each landmark time s; predict conditional survival P(T>t*|T>s, Z(s)) pooled across landmark datasets; `LandmarkModel { s_seq, max_horizon }`
- [x] `aft/restricted_spline.rs` — Restricted cubic spline baseline hazard (Royston-Parmar 2002 extended): natural cubic splines on log(-log(S(t))) with boundary + interior knots; smooth and monotone hazard without piecewise assumption; `RcsHazard`
- [x] `nonparametric/npsurv_bayes.rs` — Nonparametric Bayesian survival (Ferguson 1973, Hjort 1990 Beta process): Dirichlet process prior on F; posterior draws via stick-breaking truncation + KM-compatible hazard atoms; `DpSurvivalPosterior`
- [ ] `cox/causal_cox.rs` — Causal Cox model (Martinussen 2011): estimating causal hazard ratio under unmeasured confounding via instrumental variable; control-function residual approach; tie with `oxicuda-causal` IV framework
- [x] `screening/cif_sis.rs` — CIF-SIS (Sure Independence Screening for cumulative incidence, Fu 2017): marginal subdistribution-hazard ranking for variable screening in competing-risks high-dimensional data; `CifSis { threshold: f32 }`
- [x] `calibration/pseudo_r2.rs` — Royston-Sauerbrei pseudo-R² for survival models (Royston 2004): R²_D based on D statistic; separates explained randomness from baseline hazard; `PseudoR2Survival`
- [x] `rmst/milestone_analysis.rs` — Milestone analysis (Royston-Parmar 2011): at-risk event rates + RMST differences at pre-specified time milestones; suitable for immunotherapy OS trials; `MilestoneAnalysis { milestones: Vec<f32> }`

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No GPU runtime dependency at the source level: PTX kernels are emitted as strings; downstream Vol.1-2 (`oxicuda-driver`, `oxicuda-launch`, `oxicuda-ptx`) handle execution.

## Quality Status

- Warnings: 0 (clippy clean, `-D warnings` all-targets)
- Tests: 785 passing
- unwrap() calls: 0 (production code)
- `#![forbid(unsafe_code)]` at crate root
- Pure Rust: no C/C++/Fortran in default features

## Performance Targets

Representative algorithmic benchmarks (CPU-side reference + PTX generation timing):

| Routine | Problem size | Priority |
|---------|--------------|----------|
| Kaplan-Meier | n in {1e3, 1e5} | High |
| Cox Newton-Raphson | (n, p) in {(1e3, 5), (1e4, 20)} | High |
| Log-rank test | n in {1e3, 1e4}, k = 2-5 | High |
| RMST integration | n in {1e3, 1e5} | High |
| Schoenfeld residuals | (n, p) in {(1e3, 5), (1e4, 20)} | Mid |
| Fine-Gray IPCW | (n, p) = (1e4, 10), 2 competing causes | Mid |
| AFT Weibull MLE | n in {1e3, 1e4} | Mid |
| Concordance C | n in {1e3, 1e4} | Mid |

Target for GPU execution path: match `lifelines` / `survival` numerical agreement
within 5% (beta) / 1% (S(t)) and outperform CPU at n >= 1e4 once `oxicuda-launch`
orchestrates the emitted PTX on Linux + NVIDIA.

## Notes

- All routines accept right-censored data as the default; left-truncation will land
  with the Truncation extension (P1).
- Cox partial likelihood uses log-sum-exp stabilisation in risk-set summation.
- Newton-Raphson includes Armijo step-halving to avoid divergence on flat likelihoods.
- Greenwood variance uses the log-log transform for proper CI coverage near S(t) ~ 0 or 1.
- Fine-Gray weights are stored once per dataset and reused across iterations.

---

## Architecture-Specific Deepening

### PTX Coverage Matrix

| Kernel | sm_70 | sm_75 | sm_80 | sm_86 | sm_89 | sm_90 |
|--------|-------|-------|-------|-------|-------|-------|
| `km_step` | [x] | [x] | [x] | [x] | [x] | [x] |
| `cox_risk_sum` | [x] | [x] | [x] | [x] | [x] | [x] |
| `cox_score` | [x] | [x] | [x] | [x] | [x] | [x] |
| `cox_info` | [x] | [x] | [x] | [x] | [x] | [x] |
| `logrank_oe` | [x] | [x] | [x] | [x] | [x] | [x] |
| `brier_score` | [x] | [x] | [x] | [x] | [x] | [x] |
| `rmst_integrate` | [x] | [x] | [x] | [x] | [x] | [x] |

All six SM versions produce non-empty PTX strings and pass content-substring checks in `e2e_tests.rs`.

### Per-Architecture Optimisation Hooks
- [ ] sm_80 (Ampere) -- warp-shuffle prefix-sum for `cox_risk_sum` across the sorted risk set
- [ ] sm_89 (Ada) -- mixed-precision FP16 accumulation for `cox_info` when (n, p) is huge
- [ ] sm_90 (Hopper) -- TMA-loaded covariate tiles for `cox_score` outer product
- [ ] Verify `rmst_integrate` rectangular vs. trapezoidal accuracy on all SM versions

---

## Deepening Opportunities

### Verification Gaps (require Linux + NVIDIA hardware)
- [ ] GPU run of all 7 PTX kernels under `cargo nextest --features gpu-tests` on sm_80 / sm_89 / sm_90
- [ ] Cross-validation against `lifelines` / R-`survival` on canonical datasets (Veterans, NWTCO, GBSG2)
- [ ] Throughput vs. CPU `lifelines` at n = 1e5 with 50 covariates

### Algorithmic Deepening
- [x] Backtracking + Wolfe line search (`cox/line_search.rs` -- Armijo + strong Wolfe with cubic zoom)
- [x] Trust-region Newton for ill-conditioned Fisher information (`cox/trust_region.rs` -- Steihaug-CG)
- [x] Coordinate descent over a regularisation path (warm starts) for penalised Cox (`cox/penalised_cox.rs`)
- [x] Time-dependent Cox with delayed entry and concomitant covariate updates (`cox/time_dep_cox.rs`)
- [x] Stratified Cox with strata-specific baseline hazards (`cox/stratified_cox.rs`)

### API Polish
- [x] Builder-style `CoxBuilder::ties(TieMethod::Efron).max_iter(50).tolerance(1e-6).fit(&dataset)` (`cox/cox_builder.rs`)
- [x] Predict trait: `predict_risk`, `predict_log_hazard` for fitted models (`cox/cox_builder.rs::CoxFitResult`)
- [x] Plot-friendly helpers that emit step-function arrays for KM / NA / CIF (`plot/step_functions.rs`)
- [x] Proportional-hazards LR test, Wald test, Score test (`test/ph_lr_test.rs`)
  (for penalised Cox path via FISTA)
