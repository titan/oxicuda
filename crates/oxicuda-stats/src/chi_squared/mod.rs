//! Chi-squared and related categorical tests.

pub mod chi2_independence;
pub mod fisher_exact;
pub mod fisher_exact_fast;
pub mod mcnemar;

pub use chi2_independence::{Chi2IndependenceResult, chi2_independence};
pub use fisher_exact::{FisherExactResult, fisher_exact_2x2};
pub use fisher_exact_fast::{Alternative, fisher_exact_fast, log_choose, midp_fisher_exact_fast};
pub use mcnemar::{McnemarResult, mcnemar};
