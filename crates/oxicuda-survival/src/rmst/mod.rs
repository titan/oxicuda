//! Restricted mean survival time (RMST).

pub mod milestone_analysis;
pub mod pseudo_obs;
pub mod restricted_mean;
pub mod rmst_estimator;
pub mod trapezoid;

pub use milestone_analysis::{
    MilestoneContrast, MilestoneSummary, milestone_analysis, milestone_two_arm,
};
pub use pseudo_obs::{
    PseudoObsConfig, PseudoObsOutcome, PseudoObsRegression, PseudoObsResult, pseudo_obs_fit,
};
pub use restricted_mean::restricted_mean_from_curve;
pub use rmst_estimator::{RmstResult, rmst_from_dataset};
pub use trapezoid::{
    QuadratureComparison, compare_quadrature, rectangle_rmst_from_grid, trapezoidal_rmst_from_grid,
};
