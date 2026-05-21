//! Client selection strategies for federated learning rounds.

pub mod power_of_choice;
pub mod random;

pub use power_of_choice::{PowerOfChoice, PowerOfChoiceConfig, SelectionStrategy};
