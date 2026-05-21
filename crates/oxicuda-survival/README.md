# oxicuda-survival

Survival analysis and time-to-event modelling -- a pure Rust toolkit covering
non-parametric, semi-parametric, and parametric methods.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-survival` implements the survival-analysis stack in pure Rust, with
no external linear-algebra dependencies. Non-parametric estimation includes
Kaplan-Meier, Nelson-Aalen, life-table estimators, gamma frailty for
clustered data, multi-state models, recurrent-event MCF, net survival
(Ederer I/II, Pohar Perme), and a survival random forest. Hypothesis tests
include log-rank, stratified log-rank, Peto-Peto, and Gehan-Breslow.

The semi-parametric layer centres on Cox proportional hazards (Breslow and
Efron tie handling, Newton-Raphson, line-search and trust-region variants),
penalised Cox (ridge/lasso/elastic-net), gradient-boosted Cox, IPTW/AIPW,
stratified Cox, time-varying covariates, Schoenfeld residuals, and a mixture
cure model. Parametric AFT covers exponential, Weibull, log-normal,
log-logistic, generalised gamma, Royston-Parmar splines, and discrete-time.

Competing risks supports cumulative incidence, cause-specific Cox, and
Fine-Gray; longitudinal modelling provides joint longitudinal-survival fits.
Calibration and discrimination metrics include Brier (naive, IPCW,
integrated), time-dependent ROC, decision-curve analysis, Harrell's C, Uno's
C, and restricted mean survival time. Bayesian survival via MCMC and deep
survival heads (Cox partial-likelihood gradient for DL backends) round out
the surface, along with sample-size and power calculators
(Freedman, Schoenfeld).

## Modules

| Module | Description |
|--------|-------------|
| `data` | `Observation`, `Dataset`, `RiskSet`, truncation and counting-process primitives |
| `nonparametric` | Kaplan-Meier, Nelson-Aalen, life table, frailty, multi-state, recurrent, net survival, random forest |
| `test` | Log-rank, stratified log-rank, Peto, Gehan, score/Wald/LR tests, sample-size helpers |
| `cox` | Cox PH (Breslow/Efron, Newton-Raphson), penalised, gradient-boost, IPTW/AIPW, stratified, cure model, predictions |
| `aft` | Exponential, Weibull, log-normal, log-logistic, generalised gamma, Royston-Parmar, discrete-time AFT |
| `time_varying` | Counting-process Cox with time-varying covariates |
| `competing` | Cumulative incidence, cause-specific Cox, Fine-Gray |
| `rmst` | Restricted mean survival time and pseudo-observations regression |
| `concordance` | Harrell's C, Uno's C |
| `calibration` | Brier, IPCW Brier, integrated Brier, time-dependent ROC/AUC, DCA |
| `bayes` | Bayesian survival via Markov Chain Monte Carlo |
| `longitudinal` | Joint longitudinal-survival models |
| `deep` | DeepSurv-style head, Cox partial-likelihood loss and gradient |
| `metrics` | Median survival, RMST, S(τ) summaries |
| `plot` | Step-function helpers for KM / Nelson-Aalen / CIF plotting |
| `special` | gammaln, digamma |
| `linalg` | Internal Cholesky, Gauss-Jordan inverse, matmul helpers |
| `handle` | `SurvivalHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernel strings for survival operations |
| `error` | `SurvivalError` / `SurvivalResult` |

## Quick Start

```rust,no_run
use oxicuda_survival::data::dataset::Dataset;
use oxicuda_survival::nonparametric::kaplan_meier::kaplan_meier_estimate;
use oxicuda_survival::error::SurvivalResult;

fn main() -> SurvivalResult<()> {
    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let events = vec![true, true, false, true, true, false];
    let data = Dataset::from_arrays(&times, &events)?;

    let km = kaplan_meier_estimate(&data)?;
    for (t, s) in km.times.iter().zip(km.survival.iter()) {
        println!("t = {t:.2}  S(t) = {s:.4}");
    }
    Ok(())
}
```

## Status

**Alpha** -- 25,296 SLoC, 628 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
