//! Multiple-comparison adjustments and post-hoc tests.

pub mod bh_fdr;
pub mod bonferroni;
pub mod by_fdr;
pub mod holm;
pub mod tukey_hsd;

pub use bh_fdr::bh_fdr;
pub use bonferroni::bonferroni;
pub use by_fdr::by_fdr;
pub use holm::holm;
pub use tukey_hsd::{TukeyResult, tukey_hsd};
