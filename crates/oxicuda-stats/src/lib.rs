//! `oxicuda-stats` — Statistical inference, hypothesis testing, and frequentist analysis for OxiCUDA.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-stats
//! ├── special/         — Special functions: erf, lgamma, digamma, betainc, gammp
//! ├── distributions/   — Normal, Student-t, chi-squared, F, beta, gamma, binomial, Poisson, exponential
//! ├── descriptive/     — Mean, variance, robust statistics, quantiles
//! ├── parametric/      — t-tests, ANOVA, MANOVA, regression inference
//! ├── nonparametric/   — Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
//! ├── goodness_of_fit/ — KS, Anderson-Darling, Shapiro-Wilk, Jarque-Bera
//! ├── chi_squared/     — Chi-squared independence, Fisher exact, McNemar
//! ├── multiple/        — Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli, Tukey HSD
//! ├── resampling/      — Bootstrap, jackknife, permutation tests
//! ├── ci/              — Confidence intervals (normal, t, bootstrap, proportion)
//! ├── regression/      — OLS, ridge, logistic regression
//! ├── power/           — t-test, ANOVA power analysis and effect sizes
//! └── correlation/     — Pearson, Spearman, Kendall's tau
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod chi_squared;
pub mod ci;
pub mod correlation;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod goodness_of_fit;
pub mod handle;
pub mod multiple;
pub mod nonparametric;
pub mod parametric;
pub mod power;
pub mod ptx_kernels;
pub mod regression;
pub mod resampling;
pub mod special;

pub use error::{StatsError, StatsResult};
pub use handle::{LcgRng, SmVersion, StatsHandle};

#[cfg(test)]
mod e2e_tests;
