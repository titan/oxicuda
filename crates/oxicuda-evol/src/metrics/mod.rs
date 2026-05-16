//! Quality metrics for multi-objective evolutionary algorithms.

pub mod metrics;

pub use metrics::{
    average_nn_distance, extract_pareto_front, generational_distance, hypervolume_2d, igd, spacing,
};
