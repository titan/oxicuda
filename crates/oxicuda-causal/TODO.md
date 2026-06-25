# oxicuda-causal TODO

GPU-accelerated causal-inference primitives, covering DAG representation, causal
discovery, treatment-effect estimation, instrumental variables, causal forests,
twin-network counterfactuals, do-calculus identification, and causal evaluation
metrics.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.41).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~27,800 total lines (89 files; +9 `verification/` modules)
- **Coverage:** DAG with cycle-safe add/remove and Kahn topological sort;
  d-separation via Bayes-ball with collider handling; NOTEARS linear SEM via
  augmented Lagrangian with Padé-(3,3) matrix-exponential acyclicity;
  PC algorithm Fisher-Z skeleton + Meek orientation rules (R1–R4); GES BIC-scored
  greedy equivalence search; NOTEARS-MLP column-norm acyclicity; propensity
  logistic regression (GD); IPW ATE / ATT (clipped propensity); S / T / X-learners
  over OLS base; AIPW doubly robust ATE; Chernozhukov Double-ML (K-fold
  cross-fitting); DragonNet shared representation + 3 heads + targeted
  regularization; 2SLS instrumental variable; DeepIV two-stage MLP; causal forest
  with honest splitting (Wager-Athey 2018); twin-network counterfactual; do-calculus
  backdoor / frontdoor admissibility + adjustment-set search; and PTX kernel-string
  generation for 6 SM tiers.

### Completed

#### Core Infrastructure
- [x] error.rs — `CausalError`, `CausalResult<T>`
- [x] handle.rs — `LcgRng` deterministic PRNG with `next_normal`,
  `SmVersion` PTX target descriptor

#### DAG & d-Separation (dag/)
- [x] dag.rs — `Dag` adjacency-matrix representation, cycle-safe BFS `add_edge`,
  Kahn `topo_sort`, `has_edge`, `remove_edge`
- [x] d_separation.rs — `d_separated` via Bayes-ball with collider handling

#### Causal Discovery (discovery/)
- [x] notears.rs — `NotearsSem` augmented Lagrangian + Padé-(3,3)
  matrix-exponential acyclicity h(W) = tr(e^{W⊙W}) − d
- [x] pc.rs — `PcAlgorithm` Fisher-Z skeleton + Meek rules R1–R4 orientation
- [x] ges.rs — `Ges` BIC-scored forward + backward greedy
- [x] notears_mlp.rs — `NotearsNlp` MLP-first-layer column-norm acyclicity

#### Treatment Effect Estimation (effect/)
- [x] propensity.rs — `PropensityModel` logistic GD with sigmoid output
- [x] ipw.rs — `ipw_ate`, `ipw_att` (propensity clipped to [0.05, 0.95])
- [x] meta_learners.rs — `SLearner`, `TLearner`, `XLearner` over OLS base
- [x] doubly_robust.rs — `aipw_ate` AIPW doubly-robust estimator
- [x] double_ml.rs — `DoubleML` Chernozhukov K-fold cross-fitting
  θ̂ = mean[(T − m̂)(Y − ĝ)] / mean[(T − m̂)²]
- [x] dragonnet.rs — `DragonNet` shared representation + μ₀ / μ₁ / π heads
  + ε targeted regularization

#### Instrumental Variables (iv/)
- [x] two_sls.rs — `TwoSls` stage-1 OLS T~Z, stage-2 OLS Y~T̂
- [x] deepiv.rs — `DeepIv` two-stage MLP with ReLU hidden layers

#### Causal Forest (forest/)
- [x] causal_forest.rs — `CausalForest` honest estimation (separate build /
  estimate samples), heterogeneous split score
  (τ_L − τ_R)² · n_L · n_R / n, random √p feature subsets

#### Counterfactual (counterfactual/)
- [x] twin_network.rs — `TwinNetwork` shared MLP encoder + dual decoder
  (factual / counterfactual reconstruction)

#### Do-Calculus (do_calculus/)
- [x] identification.rs — `backdoor_admissible` (G_x̄ mutilation = remove
  outgoing edges from X then d-sep check), `frontdoor_admissible`,
  `backdoor_adjustment` minimal-valid-set search

#### Causal Metrics (metrics/)
- [x] causal_metrics.rs — `pehe` (√MSE of CATE), `ate_bias`, `policy_risk`,
  `qini_coeff` (uplift-curve area), `r_squared_cate`

#### Verification & Numerical Accuracy (verification/)
- [x] reference.rs — independent reference numerics: erf-based `normal_cdf`
  (A&S 7.1.26), bisection `two_sided_z_quantile`, Jacobi-eigendecomposition
  `expm_symmetric_eig` matrix exponential
- [x] synthetic.rs — ground-truth data generators: `LinearSem` (linear-Gaussian
  SEM with known weighted DAG), `chain_sem`/`collider_sem`/`random_dag_sem`,
  `hetero_effect_data` (known heterogeneous CATE), `confounded_data`
  (confounded constant-effect DGP)
- [x] graph_metrics.rs — `skeleton_score` (P/R/F1), `structural_hamming_distance`,
  `orientation_accuracy`
- [x] matrix_exp.rs — Padé(1,1) vs eigendecomposition error report for the
  NOTEARS acyclicity exponential
- [x] fisher_z.rs — Fisher-Z critical-value calibration + empirical type-I error
- [x] notears_recovery.rs — NOTEARS structure recovery vs ground-truth DAGs
- [x] pc_orientation.rs — PC skeleton/v-structure correctness on benchmark motifs
- [x] dml_coverage.rs — Double-ML 95% CI coverage & standard-error study
- [x] forest_pehe.rs — causal-forest PEHE on heterogeneous-effect DGPs

> **Latent bugs fixed while writing these (production code, not tests):**
> (1) `NotearsSem::fit` short-circuited on the initial W = 0 iterate (trivially
> acyclic ⇒ `h ≈ 0`) and fit *nothing*; rewritten as a proper augmented-Lagrangian
> loop with an adaptive step that never exits before descending, plus a correct
> L1 soft-threshold prox (the old `sign·|w| − lr·λ` was not a shrinkage operator)
> and a real penalty schedule. (2) `expm_scaling_exponent` could return `s ≥ 64`,
> overflowing `1u64 << s`; now capped at 63. (3) `compute_gradient` was O(n·d³)
> (recomputing `XW` inside the i,j loop); reformulated to two O(n·d²) passes
> (≈17× faster on the recovery tests). (4) PC v-structure orientation was dead
> code — the outer guard required adjacency while the body required
> non-adjacency, so no collider was *ever* oriented; fixed to iterate unshielded
> (non-adjacent) pairs.

#### PTX Kernel Generation (ptx_kernels.rs)
- [x] 7 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `partial_corr_ptx`, `notears_loss_ptx`, `expm_pade_ptx`,
  `propensity_logit_ptx`, `ipw_estimator_ptx`, `dml_residual_ptx`,
  `causal_split_score_ptx`

#### Tests & Benchmarks
- [x] 12+ end-to-end tests in `lib.rs::e2e_tests` (DAG add / remove, cycle
  detection, d-separation chain, PC algorithm, NOTEARS acyclic fit,
  propensity in [0,1], IPW ATE finite, double-ML ATE finite,
  DragonNet forward finite, causal forest fit + predict, backdoor admissible
  chain, PTX non-empty × all SM versions)
- [x] Benchmarks (`benches/causal_ops.rs`) — PTX bench group + partial
  correlation bench + DML residual bench
- [x] 35 verification tests (`verification/*`) — matrix-exp accuracy, Fisher-Z
  calibration, NOTEARS / PC structure recovery, DML coverage, forest PEHE
- **Tests:** 781 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently
  PTX-string generation tested only)
- [ ] NOTEARS gradient kernel timed end-to-end on real GPU
- [ ] DML K-fold cross-fit residual kernel measured on real GPU

#### P1 — Causal Discovery Extensions
- [x] FCI (Fast Causal Inference) — extension of PC handling latent confounders (`discovery/fci.rs` + `discovery/fci_numeric.rs` -- Spirtes-Meek-Richardson 1999; Possible-D-Sep skeleton + Zhang 2008 R1-R4 orientation rules; PAG output with {Tail, Arrow, Circle} marks)
- [x] RFCI (`discovery/rfci.rs` + `discovery/rfci_tests.rs` -- Colombo-Maathuis-Kalisch-Richardson 2012; PC-style skeleton without Possible-D-Sep + unshielded-triple collider orientation + Zhang R1-R4)
- [x] GFCI hybrid score+constraint variant (`discovery/gfci.rs` + `discovery/gfci_tests.rs` -- Ogarrio-Spirtes-Ramsey 2016 UAI; Phase 1 BIC-scored GES skeleton + Phase 2 PAG conversion with unshielded-triple collider orientation + Zhang R1-R4 to fixpoint)
- [x] LiNGAM (linear non-Gaussian) ICA-based discovery (`discovery/lingam.rs` + `discovery/lingam_tests.rs` -- Shimizu-Hoyer-Hyvärinen-Kerminen 2006 JMLR 7:2003; FastICA deflationary fixed-point with g∈{tanh,gauss,cube}, inline cyclic Jacobi whitening, greedy row-permutation for diagonal maximization, B = I − W_scaled with Shimizu Algo 2 lower-triangular permutation)
- [x] DirectLiNGAM ordering-based variant (`discovery/direct_lingam.rs` -- Shimizu et al. 2011; kurtosis-based non-Gaussianity ordering + ridge-OLS B recovery)
- [x] DAG-GNN / DAG-NoCurl differentiable variants
- [x] CD-NOD heterogeneous-data causal discovery (discovery/cd_nod.rs -- Huang et al. 2020 JMLR; surrogate domain variable C, PC skeleton over augmented {X,C}, regression-residual Fisher-Z CI, changing-mechanism detection via C-adjacency, collider+Meek orientation)

#### P1 — Effect Estimation Extensions
- [x] R-Learner (Nie & Wager 2021) residual-on-residual loss for CATE (`effect/r_learner.rs` -- K-fold cross-fit + ridge nuisance regressions + Robinson partial-out)
- [x] Bayesian Additive Regression Trees (BART) for nonparametric outcome models (effect/bart.rs -- Chipman 2010; sum-of-M-shallow-trees ensemble via greedy backfitting on residuals (BART-light, no MCMC); shrinkage + depth/min-leaf controls)
- [x] Targeted Maximum Likelihood Estimator (TMLE) — full TMLE 2-step update (`effect/tmle.rs` + `effect/tmle_tests.rs` -- van der Laan-Rubin 2006 / Gruber-van der Laan 2010; K-fold cross-fit ridge OLS Q̂⁰ + cross-fit logistic ĝ + clever-covariate H + ε-targeting update + influence-curve SE)
- [x] G-computation (`effect/g_computation.rs` -- Robins 1986; ridge OLS on `[1, T, X, T·X]` design with ATE/ATT counterfactual prediction)
- [x] Sequential g-estimation for time-varying treatments (`effect/sequential_g.rs` -- Robins 1994 Commun Statist 23:2379; Structural Nested Mean Model g-estimating equation, logistic propensity at each time point, ConstantAdditive closed-form γ̂=num/den, LinearModifier 2×2 system, bootstrap SE)
- [x] Synthetic-control method for panel data (`effect/synthetic_control.rs` -- Abadie-Diamond-Hainmueller 2010; projected gradient on simplex weights)
- [x] Regression Discontinuity (RDD) local-linear estimator (`effect/rdd.rs` + `effect/rdd_tests.rs` -- Imbens-Kalyanaraman 2012; sharp RDD weighted local-linear with Triangular/Uniform/Epanechnikov kernels, IK plug-in optimal bandwidth, kernel-aware constant C_K, per-side residual-variance sandwich SE)

#### P1 — Instrumental Variable Extensions
- [x] Anderson-Rubin weak-IV-robust confidence sets (`iv/anderson_rubin.rs` + `iv/anderson_rubin_tests.rs` -- Anderson-Rubin 1949 / Andrews-Marmer-Yu 2019; F-form `AR(β) = ((n−q)·‖P_Z e‖²) / (q·‖M_Z e‖²)` with inline F-CDF + grid-inversion confidence set)
- [x] LATE (Local Average Treatment Effect) compliance-based estimator (`iv/late.rs` -- Imbens-Angrist 1994; Wald estimator with delta-method SE, compliance/always-taker/never-taker decomposition, monotonicity check)
- [x] Heckman two-step selection model (`iv/heckman.rs` + `iv/heckman_tests.rs` -- Heckman 1979 Econometrica; Stage 1 probit Newton-Raphson with ridge-stabilised Hessian, inverse Mills ratio λ̂ = φ(γ̂ᵀZ)/Φ(γ̂ᵀZ) via A&S 26.2.17, Stage 2 ridge OLS on `[1, X, λ̂]`, White heteroskedasticity-consistent sandwich SE, ρ̂ = λ_coef/σ̂_e clipped)
- [x] GMM / 2-step GMM with optimal weighting (`iv/gmm.rs` + `iv/gmm_tests.rs` -- Hansen 1982 Econometrica 50:1029; Stage 1 identity-weighting initial estimate, heteroskedasticity-robust optimal weight `Ŵ = ((1/n)·Σ gᵢgᵢᵀ + λI)⁻¹`, Stage 2 efficient estimate (optionally iterative continuously-updating GMM), Hansen J overidentification stat with χ²(q−p) p-value via Numerical Recipes regularized lower incomplete gamma; asymptotic variance `(1/n)·(Σ_XZ Ŵ Σ_ZX + λI)⁻¹` with sample-moment normalisation)

#### P1 — Causal Forest Extensions
- [x] Generalized Random Forests (GRF) Athey & Wager 2019 (multi-target moments) — grf.rs (gradient-forest splitting, honest subsampling, IJ variance, CausalEffect+LocalLinear moments)
- [x] Local centering pre-processing — local_centering.rs (K-fold cross-fit ridge, Robinson ATE, pseudo-CATE proxy)
- [x] Honest confidence intervals via subsampled bootstrap

#### P2 — Sensitivity & Robustness
- [x] E-value (Vanderweele & Ding 2017) sensitivity bounds (`sensitivity/e_value.rs` + new `sensitivity/` submodule -- VanderWeele-Ding 2017 Annals of Internal Medicine 167:268; closed-form `E = RR + √(RR·(RR−1))` for the four supported effect types, OR→RR via §4 formula `OR/(1−p₀+p₀·OR)` with rare-outcome shortcut, HR→RR via §S.2 `(1−0.5^√HR)/(1−0.5^√(1/HR))`, RD→RR conversion, CI bound closer to null = 1.0 if interval crosses null)
- [x] Rosenbaum bounds for matched pairs (`sensitivity/rosenbaum_bounds.rs` + `sensitivity/rosenbaum_bounds_tests.rs` -- Rosenbaum 1987 Biometrika 74:13 / Rosenbaum 2002 Observational Studies §4.4; Wilcoxon signed-rank sensitivity with Pratt zero-drop, ascending average-rank ties, A&S 7.1.26 erf-based Φ, 2^n exact enumeration for n<19 / normal-approx with continuity correction for n≥20, bisection on `[1, 20]` for critical Γ)
- [x] Manski partial-identification intervals (`sensitivity/manski_bounds.rs` -- Manski 1990 AER 80:319 / Manski-Pepper 2000 AER 90:997; sharp ATE bounds under four assumptions: NoAssumption (worst-case interval width = y_upper−y_lower), MeanIndependence (point identification), MonotoneTreatmentResponse (lower ATE ≥ 0), MonotoneTreatmentSelection (MTS tighter counterfactual imputation))
- [x] Continuous sensitivity analysis a la Cinelli & Hazlett 2020 (`sensitivity/cinelli_hazlett.rs` -- Cinelli-Hazlett 2020 JRSS-B 82:39; partial-R² OVB bounds, robustness value RV_q via quadratic formula, extreme scenario bias, grid benchmark with adjusted t-statistics)

#### P2 — Counterfactual & Mediation
- [x] CEVAE counterfactual variational autoencoder
- [x] GANITE GAN-based individualized treatment effect
- [x] Imai-Keele-Tingley causal mediation decomposition (`effect/mediation.rs` + `effect/mediation_tests.rs` -- Imai-Keele-Tingley 2010 Psychological Methods 15:309; mediator ridge OLS on `[1, t, X]`, outcome ridge OLS on `[1, t, m, t·m, X]` with T·M interaction, four counterfactual predictions Ŷ(t', M̂(t)), ACME = (δ̂(0)+δ̂(1))/2, ADE = (ζ̂(0)+ζ̂(1))/2, parametric-bootstrap Monte Carlo CIs via LcgRng-seeded coefficient resampling with nearest-rank quantiles)
- [x] Pearl mediation formula identification

#### P2 — Policy Learning
- [x] Staggered DiD Callaway-Sant'Anna (`effect/staggered_did.rs`) — Callaway-Sant'Anna 2021 JOE: heterogeneity-robust difference-in-differences for staggered adoption with group-time ATT and doubly-robust aggregation; `StaggeredDid`
- [x] Doubly-robust policy learning over CATE forests (forest/dr_policy.rs -- AIPW DR scores Γ_i(a)=m̂(X_i,a)+(Y_i−m̂)·𝟙{T_i=a}/ê(X_i,a) with propensity clipping, then welfare-maximizing PolicyTree fit; reuses policy_tree.rs)
- [x] Welfare-maximizing policy trees (Athey & Wager 2021) (forest/policy_tree.rs -- exact exhaustive shallow-tree search maximizing summed doubly-robust scores with min-leaf constraint)
- [x] Off-policy evaluation (IPS, SNIPS, doubly-robust) for bandits

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA-SDK / nvcc / DoWhy / EconML / CausalML dependency — PTX kernels are
emitted as strings. No oxicuda-driver / -memory / -launch dependency at this layer.

## Quality Status

- Warnings: 0 (clippy clean, workspace lints inherited)
- Tests: 781 passing (DAG, d-sep, NOTEARS, PC, propensity, IPW, DML, DragonNet,
  causal forest, backdoor, PTX × 6 SM, + 35 verification/numerical-accuracy,
  + 11 discrete-CI chi-square/G + PC on synthetic discrete networks)
- unwrap() calls: 0 in production code
- macOS: compiles but returns `UnsupportedPlatform` at runtime when actual launch
  is attempted (PTX emission still works on every host)
- Refactoring policy: every source file is well under 2,000 lines

## Performance Targets

| Workload | Target |
|----------|--------|
| Partial-correlation Fisher-Z (n=10⁵, d=64) | ≥ 90% of cuSOLVER reference |
| NOTEARS loss + gradient (d=128) | ≥ 85% of cuBLAS-backed reference |
| Padé-(3,3) matrix exponential (d=128) | ≥ 85% of cuBLAS-backed reference |
| Double-ML K-fold residual (n=10⁶) | memory-bandwidth bound |
| Causal-forest split-score evaluation | ≥ 80% of CPU reference (grf) |

Performance harnesses are CPU-side today; GPU-side numbers will be filled in once
the Linux+NVIDIA verification run is executed.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/causal_ops.rs`) — PTX bench group +
  partial-correlation bench + DML-residual bench

---

## Notes

- All data and parameters are FP32 today. FP64 is a future option for
  high-condition-number partial-correlation tests.
- NOTEARS Padé-(3,3) acyclicity expansion uses 6 matrix multiplications;
  larger d may benefit from Padé-(13,13) with scaling-and-squaring.
- The propensity model is single-layer logistic; multi-layer MLP propensity
  is implicit in DragonNet's shared encoder.
- The PC algorithm runs Fisher-Z (multivariate-Gaussian) for continuous data
  and a discrete chi-square / G-test (`discovery/discrete_ci.rs`) for
  categorical data; both share the `pc::ConditionalIndependenceTest` trait, so a
  custom non-parametric CI test plugs into `PcAlgorithm::run_with_test` unchanged.
- Backdoor admissibility uses G_x̄ mutilation (Pearl 2009) — outgoing edges from
  X are deleted before the d-separation check.
- `LcgRng::next_normal` uses Box-Muller transform with cached second sample.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX target string emitted for all 7 kernels
- [ ] WMMA-based covariance accumulation for partial-correlation kernel

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX target string emitted
- [ ] `cp.async` global→shared prefetch for NOTEARS gradient kernel
- [ ] Shared-memory bank-conflict-free Cholesky tile layout for Fisher-Z
- [ ] Warp-shuffle reductions for DML residual products

### Hopper (sm_90)
- [x] PTX target string emitted
- [ ] TMA-based bulk loading of design matrices for very large n
- [ ] WGMMA-based fused matrix exponential Padé pipeline

### Blackwell (sm_100)
- [x] PTX target string emitted
- [ ] Native FP4/FP6 dataset compression for high-throughput causal discovery

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage and PTX-string generation.
> These items represent the gap between current depth and full
> production-grade GPU causal inference.

### Verification Gaps
- [x] NOTEARS recovery accuracy vs. ground-truth DAGs (10, 20-node SEMs)
  (`verification/notears_recovery.rs` -- samples linear-Gaussian SEMs with known
  weighted DAGs, fits NOTEARS, thresholds W, scores skeleton F1 / SHD /
  acyclicity residual; chain recall = 1.0, 10-node recall ≥ 0.5 prec ≥ 0.6,
  20-node SHD < #true-edges). 50-node left unchecked: tractable but the
  fixed-step proximal optimizer needs many iterations at d=50 → too slow for the
  routine test gate (would need an L-BFGS inner solve / GPU).
- [x] Discrete (categorical) conditional-independence test + PC on a synthetic
  discrete Bayesian network (`discovery/discrete_ci.rs::DiscreteCiTest` --
  Pearson chi-square / G-test (likelihood-ratio = conditional mutual
  information) summed over per-stratum r_x×r_y contingency tables, adaptive
  non-empty-row/column degrees of freedom (bnlearn/Tetrad rule), p-value from a
  pure-Rust regularized upper incomplete gamma `chi_square_sf` (Numerical
  Recipes §6.2 series + Lentz continued fraction + Lanczos lnΓ, verified against
  qchisq percentiles); plugged into PC through the new
  `pc::ConditionalIndependenceTest` trait and `PcAlgorithm::run_with_test` /
  `run_discrete`. Verified: declares independence/dependence correctly including
  a *faithful* noisy collider (X⫫Y marginally but X̸⫫Y|Z = "explaining away"),
  and PC recovers the exact skeleton + collider v-structure X→Z←Y on a synthetic
  3-node collider and the exact skeleton on a 3-node chain, all data generated
  deterministically via LcgRng)
- [ ] PC orientation correctness on the *external* discrete bnlearn benchmark
  networks (Asia, Alarm, Sachs) -- data-gated (requires downloading the bnlearn
  datasets, not bundled). The discrete-CI machinery now exists (above) and is
  verified on synthetic discrete DAGs; the linear-Gaussian structural analogues
  are also verified (`verification/pc_orientation.rs` -- chain skeleton exact,
  collider v-structure orientation, 5-node random-DAG skeleton F1 ≥ 0.6, no
  false edges on independent columns)
- [x] Double-ML coverage probability (95% CI) on simulated DGPs
  (`verification/dml_coverage.rs`)
- [x] Causal-forest PEHE on simulated heterogeneous-effect DGPs
  (`verification/forest_pehe.rs` -- forest beats constant-ATE baseline, CATE
  correlation > 0.2, PEHE below effect spread)

### Implementation Deepening
> All five already existed under the P1 sections above; verified by concept,
> annotated with the real filenames.
- [x] R-Learner residual-on-residual CATE estimator (`effect/r_learner.rs`)
- [x] FCI / GFCI causal discovery with latent confounders (`discovery/fci.rs`,
  `discovery/gfci.rs`, `discovery/rfci.rs`)
- [x] LiNGAM non-Gaussian discovery (`discovery/lingam.rs`,
  `discovery/direct_lingam.rs`)
- [x] TMLE / G-computation outcome-modeling estimators (`effect/tmle.rs`,
  `effect/g_computation.rs`)
- [x] Synthetic-control panel-data estimator (`effect/synthetic_control.rs`)

### Numerical Accuracy
- [x] Padé matrix-exponential error analysis vs. eigendecomposition
  (`verification/matrix_exp.rs` + `verification/reference.rs::expm_symmetric_eig`
  -- Jacobi-eigendecomposition reference; max element-wise + trace error on
  random symmetric matrices, small-norm < 2e-3, moderate-norm rel < 5e-2)
- [x] Fisher-Z critical value calibration vs. exact percentile
  (`verification/fisher_z.rs` + `verification/reference.rs` -- erf-based normal
  CDF + bisection two-sided quantile; confirms the baked-in 1.645/1.96/2.576
  constants equal `qnorm(1−α/2)` (the `pcalg::gaussCItest` rule) and the
  empirical type-I error brackets α)
- [x] DML standard-error coverage on Monte-Carlo simulations
  (`verification/dml_coverage.rs` -- reported SE tracks empirical sd of the
  estimates within a factor of ~2, 95% CI coverage in [0.75, 1.0])

## Performance Verification Harness Status (2026-05-16)

- **PTX kernels:** harnesses at `benches/causal_ops.rs::causal_ptx`;
  CPU-side PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **Partial-correlation bench:** CPU-side timing landed.
- **DML-residual bench:** CPU-side timing landed.
