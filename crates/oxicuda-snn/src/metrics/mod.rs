//! Spike-train analysis metrics: rates, intervals, distance functions, sync,
//! avalanche criticality, information theory, and population decoding.

/// Diagnostic metrics for spike trains: rate, ISI, sync, distances.
pub mod metrics;

/// Neuronal avalanches and criticality (branching parameter, power-law MLE).
pub mod avalanche;
/// Population-vector decoding and spike-triggered average / covariance.
pub mod decoding;
/// Entropy and mutual information of spike trains via word-binning.
pub mod information;
/// Time-resolved firing-rate estimation via kernel density estimation.
pub mod kde_rate;
/// Population-coded output readout: rate decode, winner-take-all, softmax.
pub mod population_coding;

pub use avalanche::{
    Avalanche, AvalancheStats, branching_parameter, branching_parameter_global, detect_avalanches,
    powerlaw_mle_exponent,
};
pub use decoding::{
    cosine_tuning_rate, population_vector, spike_triggered_average, spike_triggered_covariance,
};
pub use information::{MiCorrection, mutual_information, spike_train_entropy};
pub use population_coding::{
    population_mean_decode, rate_decode as population_rate_decode, softmax_decode, spike_counts,
    winner_take_all,
};
