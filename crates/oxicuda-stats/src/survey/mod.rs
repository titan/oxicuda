//! Complex survey design variance estimators.
//!
//! Provides stratified, clustered, weighted, and jackknife variance estimators
//! for complex survey data following Cochran (1977).

pub mod design;

pub use design::{
    ClusterResult, StratifiedResult, SurveyDesign, cluster_variance, design_effect,
    jackknife_survey_variance, stratified_variance, weighted_mean, weighted_variance,
};
