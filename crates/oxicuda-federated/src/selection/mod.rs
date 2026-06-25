//! Client selection strategies for federated learning rounds.

pub mod fairness;
pub mod power_of_choice;
pub mod random;

pub use fairness::{
    CohortFairnessTracker, FairnessSummary, StratumMetrics, fairness_summary, jains_fairness_index,
};
pub use power_of_choice::{PowerOfChoice, PowerOfChoiceConfig, SelectionStrategy};
