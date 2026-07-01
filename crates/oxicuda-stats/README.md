# oxicuda-stats

Statistical inference, hypothesis testing, and frequentist analysis -- a pure
Rust statistics toolkit.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-stats` provides a broad statistical-analysis surface implemented
entirely in pure Rust, with no external linear-algebra dependencies. It covers
parametric and non-parametric hypothesis testing, goodness-of-fit, contingency
analysis, multiple-comparison correction, resampling-based inference,
regression (linear, ridge, logistic, robust, quantile, GLM, multinomial,
negative binomial, mixed-effects), power analysis, correlation, circular
statistics, spatial autocorrelation, survey design, and time-series tests.

Special functions (`erf`, `lgamma`, `digamma`, `betainc`, `gammp`) and a full
catalogue of probability distributions (Normal, Student-t, chi-squared, F,
beta, gamma, binomial, Poisson, exponential, ...) underpin every test, with
matching `pdf`/`cdf`/`ppf` triples. Random sampling uses the workspace
`LcgRng` (MMIX LCG) for deterministic reproducibility across runs.

## Modules

| Module | Description |
|--------|-------------|
| `special` | Special functions: erf, lgamma, digamma, betainc, gammp |
| `distributions` | Normal, Student-t, chi-squared, F, beta, gamma, binomial, Poisson, exponential |
| `descriptive` | Location, dispersion, robust measures, quantiles |
| `parametric` | t-tests, one/two-way ANOVA, MANOVA, regression inference, builder API |
| `nonparametric` | Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman |
| `goodness_of_fit` | KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera |
| `chi_squared` | Chi-squared independence, Fisher exact, McNemar |
| `multiple` | Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli, Tukey HSD |
| `resampling` | Bootstrap, jackknife, permutation tests |
| `ci` | Confidence intervals: normal, t, bootstrap, proportion |
| `regression` | OLS, ridge, logistic, GLM, robust, quantile, mixed-effects, multinomial, negative binomial |
| `power` | t-test and ANOVA power analysis; Cohen's d, Hedges' g, Glass's delta, eta-squared |
| `correlation` | Pearson, Spearman, Kendall's tau |
| `circular` | Circular mean/variance, Rayleigh test, von Mises MLE |
| `spatial` | Moran's I, Geary's C, Ripley's K |
| `survey` | Stratified, cluster, and jackknife variance under complex designs |
| `bayesian` | Conjugate updates, Bayes factors, credible intervals |
| `time_series` | ACF, Ljung-Box, Box-Pierce, ADF, KPSS, Durbin-Watson |
| `time_series_advanced` | ARCH, Chow, Bai-Perron, variance-ratio, Zivot-Andrews |
| `handle` | `StatsHandle`, `SmVersion`, `LcgRng` |
| `ptx_kernels` | GPU PTX kernel strings for statistical operations |
| `error` | `StatsError` / `StatsResult` |

## Quick Start

```rust,no_run
use oxicuda_stats::parametric::t_test::{one_sample_t, two_sample_t};
use oxicuda_stats::error::StatsResult;

fn main() -> StatsResult<()> {
    let sample = [4.9, 5.1, 5.0, 4.8, 5.2, 5.0, 4.95];
    let one = one_sample_t(&sample, 5.0)?;
    println!(
        "one-sample t = {}, df = {}, p = {}",
        one.t_statistic, one.df, one.p_value_two_sided
    );

    let a = [10.0, 12.0, 11.5, 9.5, 10.5];
    let b = [13.0, 12.5, 14.0, 13.5, 12.0];
    let two = two_sample_t(&a, &b)?;
    println!(
        "two-sample t = {}, p = {}",
        two.t_statistic, two.p_value_two_sided
    );
    Ok(())
}
```

## Status

**Alpha** -- 35,055 SLoC, 1015 passing tests. API may evolve before v1.0.

## License

Apache-2.0 -- (C) 2026 COOLJAPAN OU (Team KitaSan)
