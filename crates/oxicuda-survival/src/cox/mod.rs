//! Cox proportional hazards model and supporting routines.

pub mod baseline_hazard;
pub mod breslow_ties;
pub mod cox_ph;
pub mod efron_ties;
pub mod newton_raphson;
pub mod schoenfeld;

pub use baseline_hazard::breslow_baseline_hazard;
pub use breslow_ties::breslow_log_likelihood;
pub use cox_ph::{CoxFit, CoxPhConfig, TieMethod, fit_cox_ph};
pub use efron_ties::efron_log_likelihood;
pub use newton_raphson::newton_raphson_cox;
pub use schoenfeld::{schoenfeld_residuals, schoenfeld_test};
