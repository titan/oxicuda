//! Predictive uncertainty quantification for deep models.
//!
//! Implements the four staple post-hoc / training-time approaches to obtaining
//! distributions over predictions instead of single point estimates:
//!
//! - [`mc_dropout`] — Monte-Carlo Dropout (Gal & Ghahramani 2016) reuses
//!   training-time dropout at inference for a free-Bayes approximation.
//! - [`deep_ensemble`] — Deep Ensembles (Lakshminarayanan 2017) train M
//!   independent networks; ensemble mean is the predictive expectation,
//!   spread is epistemic uncertainty.
//! - [`swag`] — Stochastic Weight Averaging Gaussian (Maddox 2019) fits a
//!   diagonal + low-rank Gaussian posterior over weights from SGD iterates.
//! - [`laplace`] — Last-layer Laplace approximation (MacKay 1992; Daxberger 2021)
//!   builds a Gaussian posterior at the MAP using the Hessian / Fisher information.
//! - [`entropy`] — predictive entropy, mutual information (BALD), and the
//!   epistemic / aleatoric decomposition for ensemble outputs.

pub mod deep_ensemble;
pub mod entropy;
pub mod functional_laplace;
pub mod laplace;
pub mod mc_dropout;
pub mod swag;

pub use functional_laplace::{FunctionalLaplace, FunctionalLaplaceConfig};
