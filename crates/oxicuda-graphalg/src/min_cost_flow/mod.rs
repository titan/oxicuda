//! Min-cost max-flow algorithms.

pub mod successive_shortest_paths;

pub use successive_shortest_paths::{
    MinCostFlowNetwork, MinCostFlowResult, min_cost_flow_bounded, min_cost_max_flow,
};
