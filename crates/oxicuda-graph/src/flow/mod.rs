//! Flow algorithms: min-cost max-flow via Successive Shortest Paths.

pub mod min_cost_flow;

pub use min_cost_flow::{McfEdge, McfResult, min_cost_flow};
