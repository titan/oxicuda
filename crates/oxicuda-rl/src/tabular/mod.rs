//! Tabular temporal-difference control algorithms for discrete, finite MDPs.
//!
//! * [`crate::tabular::QLearning`] — off-policy TD control (Watkins 1989); the
//!   bootstrap uses `max_a' Q[s', a']`, converging to the optimal `Q*`.
//! * [`crate::tabular::Sarsa`] — on-policy TD control (Rummery & Niranjan
//!   1994); the bootstrap follows the behaviour policy's next action, with an
//!   Expected-SARSA variant that integrates over the policy distribution.
//!
//! Both agents store a dense row-major `Q[s, a]` table and update from single
//! transitions, making them ideal building blocks for small environments and
//! for unit-testing exploration / scheduling components.

pub mod q_learning;
pub mod sarsa;

pub use q_learning::QLearning;
pub use sarsa::Sarsa;
