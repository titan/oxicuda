/// AdaMerging: entropy-minimisation learning of merge coefficients (Yang et al. 2024 ICLR).
pub mod adamerging;
#[cfg(test)]
mod adamerging_tests;
/// Arithmetic helpers: DARE pruning, sign consensus, weighted sum.
pub mod arithmetic;
/// DARE-TIES: composite DARE prune + TIES sign consensus (Yu et al. 2024 ICML).
pub mod dare_ties;
/// Fisher-weighted averaging (Matena & Raffel 2022).
pub mod fisher_merging;
/// LoRA merge strategies: linear merge and TIES merging.
pub mod merge;
/// Model soups (Wortsman et al. 2022): uniform, weighted, and greedy recipes.
pub mod model_soup;
/// RegMean: closed-form regression-based merging (Jin et al. 2023).
pub mod regmean;
#[cfg(test)]
mod regmean_tests;
/// Task arithmetic: editing models via task vectors.
pub mod task_arithmetic;

pub use adamerging::{AdaMerging, AdaMergingConfig, AdaMergingResult};
pub use dare_ties::{DareTies, DareTiesConfig};
pub use fisher_merging::{FisherConfig, FisherEstimate, FisherMerging};
pub use model_soup::{ModelSoup, ModelSoupConfig};
pub use regmean::{RegMean, RegMeanConfig};
pub use task_arithmetic::{TaskArithmetic, TaskVector};
