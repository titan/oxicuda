//! Chi-squared and related categorical tests.

pub mod chi2_independence;
pub mod fisher_exact;
pub mod mcnemar;

pub use chi2_independence::{Chi2IndependenceResult, chi2_independence};
pub use fisher_exact::{FisherExactResult, fisher_exact_2x2};
pub use mcnemar::{McnemarResult, mcnemar};
